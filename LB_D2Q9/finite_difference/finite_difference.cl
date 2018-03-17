#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#define ZERO_DENSITY 1e-12


void get_BC(
    int *streamed_x,
    int *streamed_y,
    const int BC_SPECIFIER,
    const int nx,
    const int ny)
{
    if (BC_SPECIFIER == 0){ //PERIODIC
        if (*streamed_x >= nx) *streamed_x -= nx;
        if (*streamed_x < 0) *streamed_x += nx;

        if (*streamed_y >= ny) *streamed_y -= ny;
        if (*streamed_y < 0) *streamed_y += ny;
    }
    if (BC_SPECIFIER == 1){ // ZERO GRADIENT
        if (*streamed_x >= nx) *streamed_x = nx - 1;
        if (*streamed_x < 0) *streamed_x = 0;

        if (*streamed_y >= ny) *streamed_y = ny - 1;
        if (*streamed_y < 0) *streamed_y = 0;
    }
}
__kernel void
grad(
    __local double *local_fluid_1,
    __global __read_only double *input,
    __global double *output,
    const double cs,
    __constant int *cx,
    __constant int *cy,
    __constant double *w,
    const int nx, const int ny,
    const int buf_nx, const int buf_ny,
    const int halo,
    const int num_jumpers,
    const int BC_SPECIFIER
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Have to use local memory where you read in everything around you in the workgroup.
    // Otherwise, you are actually doing 9x the work of what you have to...painful.

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx_1D < buf_nx) {
        for (int row = 0; row < buf_ny; row++) {
            int temp_x = buf_corner_x + idx_1D;
            int temp_y = buf_corner_y + row;

            //Painfully deal with BC's...i.e. use periodic BC's.
            get_BC(&temp_x, &temp_y, BC_SPECIFIER, nx, ny);

            local_fluid_1[row*buf_nx + idx_1D] = rho_global[fluid_index_1*ny*nx + temp_y*nx + temp_x];
            local_fluid_2[row*buf_nx + idx_1D] = rho_global[fluid_index_2*ny*nx + temp_y*nx + temp_x];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now that all desired rhos are read in, do the multiplication
    if ((x < nx) && (y < ny)){

        double grad_x = 0;
        double grad_y = 0;

        // Get the psi at the current pixel
        const int old_2d_buf_index = buf_y*buf_nx + buf_x;

        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];
            double cur_w = w[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            double cur_field = local_fluid_1[new_2d_buf_index];


            grad_x += cur_w * cur_cx * cur_field;
            grad_y += cur_w * cur_cy * cur_field;
        }

        const int two_d_index = y*nx + x;
        int three_d_inde = fluid_index_1*ny*nx + two_d_index;
        int three_d_index_fluid_2 = fluid_index_2*ny*nx + two_d_index;

        // We need to move from *force* to force/density!
        // If rho is zero, force should be zero! That's what the books say.
        // So, just don't increment the force is rho is too small; equivalent to setting force = 0.
        Gx_global[three_d_index_fluid_1] += force_x_fluid_1;
        Gy_global[three_d_index_fluid_1] += force_y_fluid_1;

        Gx_global[three_d_index_fluid_2] += force_x_fluid_2;
        Gy_global[three_d_index_fluid_2] += force_y_fluid_2;
    }
}

__kernel void
add_interaction_force_second_belt(
    const int fluid_index_1,
    const int fluid_index_2,
    const double G_int,
    __local double *local_fluid_1,
    __local double *local_fluid_2,
    __global __read_only double *rho_global,
    __global double *Gx_global,
    __global double *Gy_global,
    const double cs,
    __constant double *pi1,
    __constant int *cx1,
    __constant int *cy1,
    const int num_jumpers_1,
    __constant double *pi2,
    __constant int *cx2,
    __constant int *cy2,
    const int num_jumpers_2,
    const int nx, const int ny,
    const int buf_nx, const int buf_ny,
    const int halo,
    const int BC_SPECIFIER,
    const int PSI_SPECIFIER,
    __constant double *parameters)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Have to use local memory where you read in everything around you in the workgroup.
    // Otherwise, you are actually doing 9x the work of what you have to...painful.

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx_1D < buf_nx) {
        for (int row = 0; row < buf_ny; row++) {
            int temp_x = buf_corner_x + idx_1D;
            int temp_y = buf_corner_y + row;

            //Painfully deal with BC's...i.e. use periodic BC's.
            get_BC(&temp_x, &temp_y, BC_SPECIFIER, nx, ny);

            local_fluid_1[row*buf_nx + idx_1D] = rho_global[fluid_index_1*ny*nx + temp_y*nx + temp_x];
            local_fluid_2[row*buf_nx + idx_1D] = rho_global[fluid_index_2*ny*nx + temp_y*nx + temp_x];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now that all desired rhos are read in, do the multiplication
    if ((x < nx) && (y < ny)){

        //Remember, this is force PER DENSITY to avoid problems
        double force_x_fluid_1 = 0;
        double force_y_fluid_1 = 0;

        double force_x_fluid_2 = 0;
        double force_y_fluid_2 = 0;

        // Get the psi at the current pixel
        const int old_2d_buf_index = buf_y*buf_nx + buf_x;

        double rho_1_pixel = local_fluid_1[old_2d_buf_index];
        double rho_2_pixel = local_fluid_2[old_2d_buf_index];

        double psi_1_pixel = 0;
        double psi_2_pixel = 0;

        get_psi(PSI_SPECIFIER, rho_1_pixel, rho_2_pixel, &psi_1_pixel, &psi_2_pixel, parameters);

        //Psi at other pixels

        double psi_1 = 0;
        double psi_2 = 0;

        for(int jump_id = 0; jump_id < num_jumpers_1; jump_id++){
            int cur_cx = cx1[jump_id];
            int cur_cy = cy1[jump_id];
            double cur_w = pi1[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            double cur_rho_1 = local_fluid_1[new_2d_buf_index];
            double cur_rho_2 = local_fluid_2[new_2d_buf_index];

            get_psi(PSI_SPECIFIER, cur_rho_1, cur_rho_2, &psi_1, &psi_2, parameters);

            force_x_fluid_1 += cur_w * cur_cx * psi_2;
            force_y_fluid_1 += cur_w * cur_cy * psi_2;

            force_x_fluid_2 += cur_w * cur_cx * psi_1;
            force_y_fluid_2 += cur_w * cur_cy * psi_1;
        }

        for(int jump_id = 0; jump_id < num_jumpers_2; jump_id++){
            int cur_cx = cx2[jump_id];
            int cur_cy = cy2[jump_id];
            double cur_w = pi2[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            double cur_rho_1 = local_fluid_1[new_2d_buf_index];
            double cur_rho_2 = local_fluid_2[new_2d_buf_index];

            get_psi(PSI_SPECIFIER, cur_rho_1, cur_rho_2, &psi_1, &psi_2, parameters);

            force_x_fluid_1 += cur_w * cur_cx * psi_2;
            force_y_fluid_1 += cur_w * cur_cy * psi_2;

            force_x_fluid_2 += cur_w * cur_cx * psi_1;
            force_y_fluid_2 += cur_w * cur_cy * psi_1;
        }

        force_x_fluid_1 *= -(G_int*psi_1_pixel);
        force_y_fluid_1 *= -(G_int*psi_1_pixel);

        force_x_fluid_2 *= -(G_int*psi_2_pixel);
        force_y_fluid_2 *= -(G_int*psi_2_pixel);

        const int two_d_index = y*nx + x;
        int three_d_index_fluid_1 = fluid_index_1*ny*nx + two_d_index;
        int three_d_index_fluid_2 = fluid_index_2*ny*nx + two_d_index;

        // We need to move from *force* to force/density!
        // If rho is zero, force should be zero! That's what the books say.
        // So, just don't increment the force is rho is too small; equivalent to setting force = 0.
        Gx_global[three_d_index_fluid_1] += force_x_fluid_1;
        Gy_global[three_d_index_fluid_1] += force_y_fluid_1;

        Gx_global[three_d_index_fluid_2] += force_x_fluid_2;
        Gy_global[three_d_index_fluid_2] += force_y_fluid_2;
    }
}
