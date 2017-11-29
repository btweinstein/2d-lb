import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import pyopencl.tools
import pyopencl.clrandom
import pyopencl.array
import ctypes as ct
import matplotlib.pyplot as plt

# Required to draw obstacles
import skimage as ski
import skimage.draw

# Get path to *this* file. Necessary when reading in opencl code.
full_path = os.path.realpath(__file__)
file_dir = os.path.dirname(full_path)
parent_dir = os.path.dirname(file_dir)

# Required for allocating local memory
float_size = ct.sizeof(ct.c_float)


def get_divisible_global(global_size, local_size):
    """
    Given a desired global size and a specified local size, return the smallest global
    size that the local size fits into. Required when specifying arbitrary local
    workgroup sizes.

    :param global_size: A tuple of the global size, i.e. (x, y, z)
    :param local_size:  A tuple of the local size, i.e. (lx, ly, lz)
    :return: The smallest global size that the local size fits into.
    """
    new_size = []
    for cur_global, cur_local in zip(global_size, local_size):
        remainder = cur_global % cur_local
        if remainder == 0:
            new_size.append(cur_global)
        else:
            new_size.append(cur_global + cur_local - remainder)
    return tuple(new_size)

class Pourous_Media(object):

    def __init__(self, sim, field_index, nu_e = 1.0, epsilon = 1.0, nu_fluid=1.0, K=1.0, Fe=1.0):

        self.sim = sim # TODO: MAKE THIS A WEAKREF

        self.field_index = np.int32(field_index)

        self.nu_e = np.float32(nu_e)
        self.epsilon = np.float32(epsilon)
        self.nu_fluid = np.float32(nu_fluid)
        self.K = np.float32(K)
        self.Fe = np.float32(Fe)

        # Determine the viscosity
        self.lb_nu_e = self.nu_e * (sim.delta_t / sim.delta_x ** 2)
        self.tau = np.float32(.5 + self.lb_nu_e / (sim.cs**2))
        self.omega = np.float32(self.tau ** -1.)  # The relaxation time of the jumpers in the simulation
        print 'omega', self.omega
        assert self.omega < 2.

        # Total force INCLUDING drag forces & body force G
        self.Fx = None
        self.Fy = None


    def initialize(self, u_arr, v_arr, rho_arr, Gx_arr=None, Gy_arr=None, f_amp = 0.0):
        """
        User passes in the u field. As density is fixed at a constant (incompressibility), we solve for the appropriate
        distribution functions.
        """

        #### VELOCITY ####
        u_host = self.sim.u.get()
        v_host = self.sim.v.get()

        u_host[:, :, self.field_index] = u_arr
        v_host[:, :, self.field_index] = v_arr

        #TODO: FOR MULTIPHASE FLOWS, WE NEED TO INITIALIZE UPRIME, VPRIME?

        self.sim.u = cl.array.to_device(self.sim.queue, u_host)
        self.sim.v = cl.array.to_device(self.sim.queue, v_host)

        #### DENSITY #####
        rho_host = self.sim.rho.get()

        rho_host[:, :, self.field_index] = rho_arr
        self.sim.rho = cl.array.to_device(self.sim.queue, rho_host)

        #### External BODY FORCES ####

        if (Gx_arr is not None) and (Gy_arr is not None):
            Gx_host = self.sim.Gx.get()
            Gy_host = self.sim.Gy.get()

            Gx_host[:, :, self.field_index] = Gx_arr
            Gy_host[:, :, self.field_index] = Gy_arr

            self.sim.Gx = cl.array.to_device(self.sim.queue, Gx_host)
            self.sim.Gy = cl.array.to_device(self.sim.queue, Gy_host)

        #### TOTAL FORCE ####
        Fx_host = np.zeros((self.sim.nx, self.sim.ny), dtype=np.float32, order='F')
        Fy_host = np.zeros((self.sim.nx, self.sim.ny), dtype=np.float32, order='F')

        self.Fx = cl.array.to_device(self.sim.queue, Fx_host)
        self.Fy = cl.array.to_device(self.sim.queue, Fy_host)

        self.update_forces()

        #### UPDATE HOPPERS ####
        self.update_feq() # Based on the hydrodynamic fields, create feq

        # Now initialize the nonequilibrium f
        # In order to stream in parallel without communication between workgroups, we need two buffers (as far as the
        # authors can see at least). f will be the usual field of hopping particles and f_temporary will be the field
        # after the particles have streamed.

        self.init_pop(amplitude=f_amp) # Based on feq, create the hopping non-equilibrium fields


    def init_pop(self, amplitude=0.001):
        """Based on feq, create the initial population of jumpers."""

        nx = self.sim.nx
        ny = self.sim.ny

        # For simplicity, copy feq to the local host, where you can make a copy. There is probably a better way to do this.
        f_host = self.sim.feq.get()
        cur_f = f_host[:, :, self.field_index, :]

        # We now slightly perturb f. This is actually dangerous, as concentration can grow exponentially fast
        # from sall fluctuations. Sooo...be careful.
        perturb = (1. + amplitude * np.random.randn(nx, ny, self.sim.num_jumpers))
        cur_f *= perturb

        # Now send f to the GPU
        f_host[:, :, self.field_index, :] = cur_f
        self.sim.f = cl.array.to_device(self.sim.queue, f_host)

    def update_forces(self):
        """
        Based on the hydrodynamic fields, create the local equilibrium feq that the jumpers f will relax to.
        Implemented in OpenCL.
        """

        sim = self.sim

        self.sim.kernels.update_forces_pourous(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.u.data, sim.v.data,
            self.Fx.data, self.Fy.data,
            sim.Gx.data, sim.Gy.data,
            self.epsilon, self.nu_fluid, self.Fe, self.K,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations
        ).wait()

    def update_feq(self):
        """
        Based on the hydrodynamic fields, create the local equilibrium feq that the jumpers f will relax to.
        Implemented in OpenCL.
        """

        sim = self.sim

        self.sim.kernels.update_feq_pourous(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.feq.data,
            sim.rho.data,
            sim.u.data, sim.v.data,
            self.epsilon,
            sim.w, sim.cx, sim.cy, sim.cs,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers).wait()

    def move_bcs(self):
        """
        Enforce boundary conditions and move the jumpers on the boundaries. Generally extremely painful.
        Implemented in OpenCL.
        """
        pass # Implemented in move_periodic in this case...it's just easier

    def move(self):
        """
        Move all other jumpers than those on the boundary. Implemented in OpenCL. Consists of two steps:
        streaming f into a new buffer, and then copying that new buffer onto f. We could not think of a way to stream
        in parallel without copying the temporary buffer back onto f.
        """

        sim = self.sim

        self.sim.kernels.move_periodic(sim.queue, sim.two_d_global_size, sim.two_d_local_size,
                                sim.f.data, sim.f_streamed.data,
                                sim.cx, sim.cy,
                                sim.nx, sim.ny,
                                self.field_index, sim.num_populations).wait()

        # Copy the streamed buffer into f so that it is correctly updated.
        cl.enqueue_copy(sim.queue, sim.f.data, sim.f_streamed.data)

    def update_hydro(self):
        """
        Based on the new positions of the jumpers, update the hydrodynamic variables. Implemented in OpenCL.
        Requires u_prime to have been updated first!
        """

        sim = self.sim

        sim.kernels.update_hydro_pourous(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f.data,
            sim.rho.data,
            sim.u_prime.data, sim.v_prime.data,
            sim.u.data, sim.v.data,
            sim.Gx.data, sim.Gy.data,
            self.epsilon, self.nu_fluid, self.Fe, self.K,
            sim.w, sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers, self.sim.delta_t
        ).wait()

        if sim.check_max_ulb:
            max_ulb = cl.array.max((sim.u**2 + sim.v**2)**.5, queue=self.queue)

            if max_ulb > sim.cs*sim.mach_tolerance:
                print 'max_ulb is greater than cs/10! Ma=', max_ulb/sim.cs

    def collide_particles(self):
        sim = self.sim

        self.sim.kernels.collide_particles_pourous(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f.data,
            sim.feq.data,
            sim.rho.data,
            sim.u.data, sim.v.data,
            self.Fx.data, self.Fy.data,
            self.epsilon, self.omega,
            sim.w, sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers, sim.delta_t, sim.cs
        ).wait()

class Simulation_Runner(object):
    """
    Everything is in dimensionless units. It's just easier.
    """

    def __init__(self, Lx=1.0, Ly=1.0,
                 time_prefactor=1., N=10, num_populations=1,
                 two_d_local_size=(32,32), use_interop=False,
                 check_max_ulb=False, mach_tolerance=0.1):
        """
        :param N: Resolution of the simulation. As N increases, the simulation should become more accurate. N determines
                  how many grid points the characteristic length scale is discretized into
        :param time_prefactor: In order for a simulation to be accurate, in general, the dimensionless
                               space discretization delta_t ~ delta_x^2 (see http://wiki.palabos.org/_media/howtos:lbunits.pdf).
                               In our simulation, delta_t = time_prefactor * delta_x^2. delta_x is determined automatically
                               by N.
        :param two_d_local_size: A tuple of the local size to be used in 2d, i.e. (32, 32)
        """

        # Dimensionless units
        self.Lx = Lx
        self.Ly = Ly

        # Book-keeping
        self.num_populations = np.int32(num_populations)

        self.check_max_ulb = check_max_ulb
        self.mach_tolerance = mach_tolerance

        # Get the characteristic length and time scales for the flow.
        self.L = 1.0 # mm
        self.T = 1.0 # Time in generations

        # Initialize the lattice to simulate on; see http://wiki.palabos.org/_media/howtos:lbunits.pdf
        self.N = N # Characteristic length is broken into N pieces
        self.delta_x = np.float32(1./N) # How many squares characteristic length is broken into
        self.delta_t = np.float32(time_prefactor * self.delta_x**2) # How many time iterations until the characteristic time, should be ~ \delta x^2

        # Characteristic LB speed corresponding to dimensionless speed of 1. Must be MUCH smaller than cs = .57 or so.
        self.ulb = self.delta_t/self.delta_x
        print 'u_lb:', self.ulb

        # Initialize grid dimensions
        self.nx = None # Number of grid points in the x direction with the boundray
        self.ny = None # Number of grid points in the y direction with the boundary
        self.initialize_grid_dims()

        # Create global & local sizes appropriately
        self.two_d_local_size = two_d_local_size        # The local size to be used for 2-d workgroups
        self.two_d_global_size = get_divisible_global((self.nx, self.ny), self.two_d_local_size)

        print '2d global:' , self.two_d_global_size
        print '2d local:' , self.two_d_local_size

        # Initialize the opencl environment
        self.context = None     # The pyOpenCL context
        self.queue = None       # The queue used to issue commands to the desired device
        self.kernels = None     # Compiled OpenCL kernels
        self.use_interop = use_interop
        self.init_opencl()      # Initializes all items required to run OpenCL code

        # Allocate constants & local memory for opencl
        self.w = None
        self.cx = None
        self.cy = None
        self.cs = None
        self.num_jumpers = None

        self.halo = None
        self.buf_nx = None
        self.buf_ny = None
        self.psi_local = None

        self.allocate_constants()

        ## Initialize hydrodynamic variables & Shan-chen variables

        rho_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=np.float32, order='F')
        self.rho = cl.array.to_device(self.queue, rho_host)

        u_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=np.float32, order='F')
        v_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=np.float32, order='F')
        self.u = cl.array.to_device(self.queue, u_host) # Velocity in the x direction; one per sim!
        self.v = cl.array.to_device(self.queue, v_host) # Velocity in the y direction; one per sim.

        u_prime_host = np.zeros((self.nx, self.ny), dtype=np.float32, order='F')
        v_prime_host = np.zeros((self.nx, self.ny), dtype=np.float32, order='F')
        self.u_prime = cl.array.to_device(self.queue, u_prime_host)  # Velocity in the x direction; one per sim!
        self.v_prime = cl.array.to_device(self.queue, v_prime_host)  # Velocity in the y direction; one per sim.

        # Intitialize the underlying feq equilibrium field
        feq_host = np.zeros((self.nx, self.ny, self.num_populations, self.num_jumpers), dtype=np.float32, order='F')
        self.feq = cl.array.to_device(self.queue, feq_host)

        f_host = np.zeros((self.nx, self.ny, self.num_populations, self.num_jumpers), dtype=np.float32, order='F')
        self.f = cl.array.to_device(self.queue, f_host)
        self.f_streamed = self.f.copy()

        # Initialize G: the body force acting on each phase
        Gx_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=np.float32, order='F')
        Gy_host = np.zeros((self.nx, self.ny, self.num_populations), dtype=np.float32, order='F')
        self.Gx = cl.array.to_device(self.queue, Gx_host)
        self.Gy = cl.array.to_device(self.queue, Gy_host)

        #### COORDINATE SYSTEM: FOR CHECKING SIMULATIONS ####

        self.x_center = None
        self.y_center = None
        self.X_dim = None
        self.Y_dim = None

        self.x_center = self.nx / 2
        self.y_center = self.ny / 2

        xvalues = np.arange(self.nx)
        yvalues = np.arange(self.ny)
        Y, X = np.meshgrid(yvalues, xvalues)
        X = X.astype(np.float)
        Y = Y.astype(np.float)

        deltaX = X - self.x_center
        deltaY = Y - self.y_center

        # Convert to dimensionless coordinates
        self.X = deltaX / self.N
        self.Y = deltaY / self.N

        # Create list corresponding to all of the different fluids
        self.fluid_list = []
        self.tau_arr = []

    def add_fluid(self, fluid):
        self.fluid_list.append(fluid)

    def complete_setup(self):
        # Run once all fluids have been added...gathers necessary info about the fluids

        # Generate the list of all relaxation times. Necessary to calculate
        # u and v prime.
        tau_host = []
        for cur_fluid in self.fluid_list:
            tau_host.append(cur_fluid.tau)
        tau_host = np.array(tau_host, dtype=np.float32)
        self.tau_arr = cl.Buffer(self.context, cl.mem_flags.READ_ONLY |
        cl.mem_flags.COPY_HOST_PTR, hostbuf=tau_host)

        # Now calculate u and v prime
        self.update_velocity_prime()

    def update_velocity_prime(self):
        self.kernels.update_velocity_prime(
            self.queue, self.two_d_global_size, self.two_d_local_size,
            self.u_prime.data, self.v_prime.data,
            self.rho.data,
            self.f.data,
            self.tau_arr,
            self.w, self.cx, self.cy,
            self.nx, self.ny,
            self.num_populations, self.num_jumpers
        ).wait()


    def initialize_grid_dims(self):
        """
        Initializes the dimensions of the grid that the simulation will take place in. The size of the grid
        will depend on both the physical geometry of the input system and the desired resolution N.
        """
        self.nx = np.int32(np.round(self.N*self.Lx))
        self.ny = np.int32(np.round(self.N*self.Ly))

        print 'nx:' , self.nx
        print 'ny:', self.ny

    def init_opencl(self):
        """
        Initializes the base items needed to run OpenCL code.
        """

        # Startup script shamelessly taken from CS205 homework
        platforms = cl.get_platforms()
        print 'The platforms detected are:'
        print '---------------------------'
        for platform in platforms:
            print platform.name, platform.vendor, 'version:', platform.version

        # List devices in each platform
        for platform in platforms:
            print 'The devices detected on platform', platform.name, 'are:'
            print '---------------------------'
            for device in platform.get_devices():
                print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
                print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
                print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
                print 'Maximum work group size', device.max_work_group_size
                print 'Maximum work item dimensions', device.max_work_item_dimensions
                print 'Maximum work item size', device.max_work_item_sizes
                print '---------------------------'

        # Create a context with all the devices
        devices = platforms[0].get_devices()
        if not self.use_interop:
            self.context = cl.Context(devices)
        else:
            self.context = cl.Context(properties=[(cl.context_properties.PLATFORM, platforms[0])]
                                                 + cl.tools.get_gl_sharing_context_properties(),
                                      devices= devices)
        print 'This context is associated with ', len(self.context.devices), 'devices'

        # Create a simple queue
        self.queue = cl.CommandQueue(self.context, self.context.devices[0],
                                     properties=cl.command_queue_properties.PROFILING_ENABLE)
        # Compile our OpenCL code
        self.kernels = cl.Program(self.context, open(file_dir + '/single_component.cl').read()).build(options='')

    def allocate_constants(self):
        """
        Allocates constants and local memory to be used by OpenCL.
        """

        ##########################
        ##### D2Q9 parameters ####
        ##########################
        w = np.array([4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36.,
                      1. / 36., 1. / 36., 1. / 36.], order='F', dtype=np.float32)  # weights for directions
        cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], order='F', dtype=np.int32)  # direction vector for the x direction
        cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], order='F', dtype=np.int32)  # direction vector for the y direction
        self.cs = np.float32(1. / np.sqrt(3))  # Speed of sound on the lattice

        self.num_jumpers = np.int32(9)  # Number of jumpers for the D2Q9 lattice: 9

        self.w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
        self.cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx)
        self.cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy)

        # Allocate local memory for the clumpiness
        self.halo = np.int32(1) # As we are doing D2Q9, we have a halo of one
        self.buf_nx = np.int32(self.two_d_local_size[0] + 2 * self.halo)
        self.buf_ny = np.int32(self.two_d_local_size[1] + 2 * self.halo)
        self.psi_local = cl.LocalMemory(float_size * self.buf_nx * self.buf_ny)

    def run(self, num_iterations, debug=False):
        """
        Run the simulation for num_iterations. Be aware that the same number of iterations does not correspond
        to the same non-dimensional time passing, as delta_t, the time discretization, will change depending on
        your resolution.

        :param num_iterations: The number of iterations to run
        """
        for cur_iteration in range(num_iterations):
            for cur_fluid in self.fluid_list:
                if debug:
                    print 'At beginning of iteration:'
                    self.check_fields()

                cur_fluid.move() # Move all jumpers
                if debug:
                    print 'After move:'
                    self.check_fields()

                cur_fluid.move_bcs() # Our BC's rely on streaming before applying the BC, actually
                if debug:
                    print 'After move bcs'
                    self.check_fields()
            self.update_velocity_prime()
            # Update forces here as appropriate
            for cur_fluid in self.fluid_list:
                cur_fluid.update_hydro() # Update the hydrodynamic variables
                if debug:
                    print 'After updating hydro'
                    self.check_fields()

                cur_fluid.update_feq() # Update the equilibrium fields
                if debug:
                    print 'After updating feq'
                    self.check_fields()

                cur_fluid.collide_particles() # Relax the nonequilibrium fields.
                if debug:
                    print 'After colliding particles'
                    self.check_fields()

    def check_fields(self):
        # Start with rho
        for i in range(self.num_populations):
            print 'Field:', i
            print 'rho_sum', cl.array.sum(self.rho[:, :, i])
            print 'f_sum', np.sum(self.f.get()[:, :, i, :])
            print 'f_eq_sum', np.sum(self.feq.get()[:, :, i, :])

        print
