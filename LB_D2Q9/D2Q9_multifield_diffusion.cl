__kernel void
update_feq(__global __write_only float *feq_global,
           __global __read_only float *rho_global,
           __global __read_only float *u_global,
           __global __read_only float *v_global,
           __constant float *w,
           __constant int *cx,
           __constant int *cy,
           const float cs,
           const int nx, const int ny, const int num_populations)
{
    //Input should be a 2d workgroup. But, we loop over a 4d array...
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int two_d_index = y*nx + x;

    if ((x < nx) && (y < ny)){

        const float u = u_global[two_d_index];
        const float v = v_global[two_d_index];

        // rho is three-dimensional now...have to loop over every field.
        for (int field_num=0; field_num < num_populations + 1; field_num++){
            int three_d_index = field_num*nx*ny + two_d_index;
            float rho = rho_global[three_d_index];
            // Now loop over every jumper
            for(int jump_id=0; jump_id < 9; jump_id++){
                int four_d_index = jump_id*field_num*nx*ny + three_d_index;

                float cur_w = w[jump_id];
                int cur_cx = cx[jump_id];
                int cur_cy = cy[jump_id];

                float cur_c_dot_u = cur_cx*u + cur_cy*v;

                float new_feq = cur_w*rho*(1.f + cur_c_dot_u/(cs*cs));

                feq_global[three_d_index] = new_feq;
            }
        }
    }
}


__kernel void
update_hydro(__global float *f_global,
             __global float *u_global,
             __global float *v_global,
             __global float *rho_global,
             const int nx, const int ny, const int num_populations)
{
    // Assumes that u and v are imposed. Can be changed later.
    //This *MUST* be run after move_bc's, as that takes care of BC's
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        // Loop over fields.
        for(int field_num = 0; field_num < num_populations + 1, field_num++){
            int three_d_index = field_num*nx*ny + two_d_index;

            float f_sum = 0;
            for(int jump_id = 0; jump_id < 9; jump_id++){
                f_sum += f_global[jump_id*field_num*nx*ny + three_d_index];
            }
            rho_global[three_d_index] = f_sum;
        }
    }
}

__kernel void
collide_particles(__global float *f_global,
                         __global float *feq_global,
                         const float omega,
                         const int nx, const int ny)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){

        const int two_d_index = y*nx + x;

        for(int jump_id = 0; jump_id < 9; jump_id++){
            int three_d_index = jump_id*nx*ny + two_d_index;

            float f = f_global[three_d_index];
            float feq = feq_global[three_d_index];

            f_global[three_d_index] = f*(1-omega) + omega*feq;
        }
    }
}



__kernel void
copy_buffer(__global __read_only float *copy_from,
            __global __write_only float *copy_to,
            const int nx, const int ny, const int num_populations)
{
    //Used to copy the streaming buffer back to the original
    //Assumes a 2d workgroup
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;

        for(int field_num=0; field_num < num_populations + 1; field_num++){
            int three_d_index = field_num*nx*ny + two_d_index;

            for (int jump_id = 0; jump_id < 9; jump_id++){

                int four_d_index = jump_id*field_num*nx*ny + three_d_index;
                copy_to[four_d_index] = copy_from[four_d_index];
            }
        }
    }
}

__kernel void
move(__global __read_only float *f_global,
     __global __write_only float *f_streamed_global,
     __constant int *cx,
     __constant int *cy,
     const int nx, const int ny, const int num_populations)
{
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        for(int jump_id = 0; jump_id < 9; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];

            //Make sure that you don't go out of the system

            int stream_x = x + cur_cx;
            int stream_y = y + cur_cy;

            if ((stream_x >= 0) && (stream_x < nx) && (stream_y >= 0) && (stream_y < ny)){ // Stream
                for(int field_num = 0; field_num < num_populations + 1; field_num++){
                    int slice = jump_id*field_num*nx*ny + field_num*nx*ny;

                    int old_4d_index = slice + y*nx + x;
                    int new_4d_index = slice + stream_y*nx + stream_x;

                    f_streamed_global[new_4d_index] = f_global[old_4d_index];
                }
            }
        }
    }
}
