// -*- C++ -*-
#ifndef N_BODY_SIM_CUDA_CUH
#define N_BODY_SIM_CUDA_CUH

#define SIMPLE 1
#define PXP 2
#define PXP_OPT 3

#define SOFT_FACTOR 10.0

void init_data(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block, int algorithm);
void init_data(int h_num_particles, float *h_particle_data, float *h_particle_vels, int h_num_blocks, int h_num_threads_per_block, int algorithm);

void delete_data();

void call_interact_kernel(float dt);

void get_particle_data(float* h_particle_data, float* h_particle_vels);

#endif // N_BODY_SIM_CUDA_CUH
