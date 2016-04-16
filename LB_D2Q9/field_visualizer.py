import numpy as np
import vispy as vp
import pyopencl as cl

field_vert_shader = """
// Uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_antialias;

// Attributes
attribute vec2 a_position;
attribute vec2 a_texcoord;

// Varyings
varying vec2 v_texcoord;

// Main
void main (void)
{
    v_texcoord = a_texcoord;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,0.0,1.0);
}
"""

field_frag_shader = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;
void main()
{
    gl_FragColor = texture2D(u_texture, v_texcoord);
    gl_FragColor.a = 1.0;
}

"""


class field_visualizer_canvas(vp.app.Canvas):

    def __init__(self, sim, sim_field_to_draw, num_steps_per_draw=1):
        # Determine the size of the window
        self.sim = sim
        self.sim_field_to_draw = sim_field_to_draw
        self.W, self.H = sim.nx, sim.ny
        vp.app.Canvas.__init__(self, keys='interactive', size=((self.W * 5), (self.H * 5)))

        # Setup necessary buffers, projections, etc.
        self.I = np.zeros((self.W, self.H), dtype=np.float32, order='F')
        cl.enqueue_copy(sim.queue, self.I, sim.u, is_blocking=True)

        # A simple texture quad. Basically, a rectangular viewing window.
        self.data = np.zeros(4, dtype=[('a_position', np.float32, 2),
                                  ('a_texcoord', np.float32, 2)])
        self.data['a_position'] = np.array([[0, 0], [self.W, 0], [0, self.H], [self.W, self.H]])
        self.data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        self.program = vp.gloo.Program(field_vert_shader, field_frag_shader)
        self.texture = vp.gloo.Texture2D(I, interpolation='linear')

        self.program['u_texture'] = self.texture
        self.program.bind(vp.gloo.VertexBuffer(self.data))

        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.projection = vp.util.transforms.ortho(0, self.W, 0, self.H, -1, 1)
        self.program['u_projection'] = self.projection

        vp.gloo.set_clear_color('white')

        self._timer = vp.app.Timer('auto', connect=self.update, start=True)

        # This must be optimized. You want the # of steps per draw that gives you about 60fps,
        # or else your simulation will go much much slower.
        self.num_steps_per_draw = num_steps_per_draw
        self.total_num_steps = 0

        self.show()

    def on_resize(self, event):
        width, height = event.physical_size
        vp.gloo.set_viewport(0, 0, width, height)
        self.projection = vp.util.transforms.ortho(0, width, 0, height, -100, 100)
        self.program['u_projection'] = self.projection

        # Compute thje new size of the quad
        r = width / float(height)
        R = self.W / float(self.H)
        if r < R:
            w, h = width, width / R
            x, y = 0, int((height - h) / 2)
        else:
            w, h = height * R, height
            x, y = int((width - w) / 2), 0
        self.data['a_position'] = np.array(
            [[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        self.program.bind(vp.gloo.VertexBuffer(self.data))

    def on_draw(self, event):
        vp.gloo.clear(color=True, depth=True)
        #I[...] = np.random.uniform(0, 1, (W, H)).astype(np.float32)
        self.sim.run(self.num_points_per_draw)
        self.num_points_run += self.num_points_per_draw
        cl.enqueue_copy(self.sim.queue, self.I, self.sim_field_to_draw, is_blocking=True)
        self.I[:, :] = self.I/np.max(self.I)
        self.texture.set_data(self.I)
        self.program.draw('triangle_strip')