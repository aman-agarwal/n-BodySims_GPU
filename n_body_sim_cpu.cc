#include <cstdio>
#include <iostream>

#include "n_body_sim_cuda.cuh"

// Flag for pingpong;
int pingpong = 0;

// Number particles; determined at runtime.
int num_particles;    
 
float* particle_vels[2]; 
float* particle_data[2]; 

void init_data(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block, int h_algorithm) 
{
  num_particles = h_num_particles;
  
  particle_vels[0] = new float[num_particles * 2];
  particle_vels[1] = new float[num_particles * 2];
 
  particle_data[0] = new float[num_particles * 3];
  particle_data[1] = new float[num_particles * 3];

  for (int i = 0; i < num_particles; i++) 
  {
    particle_vels[0][2*i] = rand() % (max_vel - min_vel) + min_vel;
    particle_vels[0][2*i + 1] = rand() % (max_vel - min_vel) + min_vel;

    particle_data[0][3*i] = rand() % box_width;
    particle_data[0][3*i + 1] = rand() % box_height;

    particle_data[0][3*i + 2] = 1;
  }
}

void delete_data() {
  // free all memory on GPU
  for (int i = 0; i < 2; i++)
  {
    delete[] particle_vels[i];
    delete[] particle_data[i];
  }
}

void add_force(int p1, int p2, float * force) {
 
  float x_dist, y_dist, dist_squared, force_magnitude;

  x_dist = particle_data[pingpong][3 * p1] - particle_data[pingpong][3 * p2];
  y_dist = particle_data[pingpong][3 * p1 + 1] - particle_data[pingpong][3 * p2 + 1];
  dist_squared = x_dist * x_dist + y_dist * y_dist + SOFT_FACTOR;

  force_magnitude = particle_data[pingpong][3 * p1 + 2] * particle_data[pingpong][3 * p2 + 2] / dist_squared;
  force[0] -= x_dist * force_magnitude / sqrt(dist_squared);
  force[1] -= y_dist * force_magnitude / sqrt(dist_squared);
}

void call_interact_kernel(float dt) {
  for (int i = 0; i < num_particles; i++)
  {
    float force[2];
    for (int j = 0; j <  num_particles; j++)
    {
      add_force(i, j, force);
    }
    particle_vels[1 - pingpong][2 * i] = particle_vels[pingpong][2 * i] + force[0] * dt / particle_data[pingpong][3 * i + 2];
    particle_vels[1 - pingpong][2 * i + 1] = particle_vels[pingpong][2 * i + 1] + force[1] * dt / particle_data[pingpong][3 * i + 2];
    
    particle_data[1 - pingpong][3 * i] = particle_data[pingpong][3 * i] + particle_vels[1 - pingpong][2 * i] * dt;
    particle_data[1 - pingpong][3 * i + 1] = particle_data[pingpong][3 * i + 1] + particle_vels[1 - pingpong][2 * i + 1] * dt;
  } 
  pingpong = 1 - pingpong;
}

void get_particle_data(float * h_particle_data, float * h_particle_vels) {
  memcpy(h_particle_data, particle_data[1 - pingpong], sizeof(float) * 3 * num_particles);
  memcpy(h_particle_vels, particle_vels[1 - pingpong], sizeof(float) * 2 * num_particles);
}
