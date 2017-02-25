#include <curand.h>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "n_body_sim_cuda.cuh"

// macro for error-handling
#define gpuErrChk(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Flag for pingpong;
int pingpong = 0;

// Number particles; determined at runtime.
int num_particles;    
 
int num_blocks;
int num_threads_per_block;

// Algorithm to use.
int algorithm;

// Device buffer variables
float2* particle_vels[2]; // x and y represent velocity in 2D
float3* particle_data[2]; // x and y represent position in 2D, z represents mass

__global__
void cudaInitKernel(float2 * vels_buffer, float3 * data_buffer1, float3 * data_buffer2, float * random, float box_width, 
                    float box_height, float min_vel, float max_vel, int num_particles)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < num_particles)
  {
    vels_buffer[i].x = min_vel + random[4 * i] * (max_vel - min_vel);
    vels_buffer[i].y = min_vel + random[4 * i + 1] * (max_vel - min_vel);
    data_buffer1[i].x = random[4 * i + 2] * box_width;
    data_buffer1[i].y = random[4 * i + 3] * box_height;
    data_buffer1[i].z = 1;

    data_buffer2[i].z = 1;    

/*    if (i == 0) {
      data_buffer1[i].z = 1000;
      data_buffer2[i].z = 1000;
      data_buffer1[i].x = box_width / 2;
      data_buffer1[i].y = box_height / 2;
      vels_buffer[i].x = 0;
      vels_buffer[i].y = 0;
    }
*/

    i += blockDim.x * gridDim.x;
  }
}

void alloc_particle_info() {
  // instantiate particle_vels, particle_data on GPU
  gpuErrChk(cudaMalloc((void **) &particle_vels[0], sizeof(float2) * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_vels[1], sizeof(float2) * num_particles));
  
  gpuErrChk(cudaMalloc((void **) &particle_data[0], sizeof(float3) * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_data[1], sizeof(float3) * num_particles));
}

void init_data(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block, int h_algorithm) 
{
  num_particles = h_num_particles;
  num_blocks = h_num_blocks;
  num_threads_per_block = h_num_threads_per_block;
  algorithm = h_algorithm;

  // instantiate particle_vels, particle_data on GPU
  alloc_particle_info();
   
  // set initial values for particle_vels, particle_data on GPU
  float * random;
  gpuErrChk(cudaMalloc((void **) &random, sizeof(float) * num_particles * 4));   
  
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateUniform(gen, random, num_particles * 4);

  cudaInitKernel<<<num_blocks, num_threads_per_block>>>(particle_vels[0], particle_data[0], particle_data[1], 
                                                        random, box_width, box_height, min_vel, max_vel, num_particles);

  curandDestroyGenerator(gen);
  gpuErrChk(cudaFree(random));
}
void init_data(int h_num_particles, float *h_particle_data, float *h_particle_vels, int h_num_blocks, int h_num_threads_per_block, int h_algorithm) {
  num_particles = h_num_particles;
  num_blocks = h_num_blocks;
  num_threads_per_block = h_num_threads_per_block;
  algorithm = h_algorithm;

  alloc_particle_info();

  gpuErrChk(cudaMemcpy(particle_data[0], h_particle_data, 3 * num_particles * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(particle_data[1], h_particle_data, 3 * num_particles * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(particle_vels[0], h_particle_vels, 2 * num_particles * sizeof(float), cudaMemcpyHostToDevice));
}

void delete_data() {
  // free all memory on GPU
  for (int i = 0; i < 2; i++)
  {
    gpuErrChk(cudaFree(particle_vels[i]));
    gpuErrChk(cudaFree(particle_data[i]));
  }
}

__device__
float2 get_force(float3 pos_data, float3 * data_old, int num_particles) {
  // sum force from every other particle based on mass, position of both particles
  float2 force;
  force.x = 0;
  force.y = 0;

  float3 other_data; // saves about 3s @ 128 threads/block and 1024 particles to store data_old[i], x_dist, and y_dist locally
  float x_dist, y_dist, dist_squared;

  float force_magnitude;
  float soft_factor = SOFT_FACTOR;
  for (int i = 0; i < num_particles; i++)
  {
    other_data = data_old[i];
    x_dist = pos_data.x - other_data.x;
    y_dist = pos_data.y - other_data.y;
    dist_squared = x_dist * x_dist + y_dist * y_dist + soft_factor;

    force_magnitude = pos_data.z * other_data.z / dist_squared;
    force.x -= x_dist * force_magnitude / sqrt(dist_squared);
    force.y -= y_dist * force_magnitude / sqrt(dist_squared);
  }
  return force;  
}

__device__
float2 get_force_opt1(float3 pos_data, float3 * data_old, int num_particles) {
  // sum force from every other particle based on mass, position of both particles
  float2 force = {0, 0};

  float3 other_data1;
  float x_dist1, y_dist1, dist_cubed1;

  float force_magnitude1;
  float soft_factor;
  for (int i = 0; i < num_particles; i+=1)
  {
    other_data1 = data_old[i];
    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;
    dist_cubed1 = pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + soft_factor, 1.5f); 

    force_magnitude1 = pos_data.z * other_data1.z / dist_cubed1; 
    force.x += x_dist1 * force_magnitude1;
    force.y += y_dist1 * force_magnitude1;   
  }
  return force;  
}

__device__
float2 get_force_opt2(float3 pos_data, float3 * data_old, int num_particles) {
  // sum force from every other particle based on mass, position of both particles
  float2 force = {0, 0};

  float3 other_data1, other_data2;
  float x_dist1, y_dist1, dist_cubed1, x_dist2, y_dist2, dist_cubed2;

  float force_magnitude1, force_magnitude2;
  float soft_factor;
  for (int i = 0; i < num_particles; i+=2)
  {
    other_data1 = data_old[i];
    other_data2 = data_old[i + 1];

    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;
    dist_cubed1 = pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + soft_factor, 1.5f); 

    force_magnitude1 = pos_data.z * other_data1.z / dist_cubed1; 
 
    x_dist2 = pos_data.x - other_data2.x;
    y_dist2 = pos_data.y - other_data2.y;
    dist_cubed2 = pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + soft_factor, 1.5f); 

    force_magnitude2 = pos_data.z * other_data2.z / dist_cubed2; 

    force.x += x_dist1 * force_magnitude1 + x_dist2 * force_magnitude2;
    force.y += y_dist1 * force_magnitude1 + y_dist2 * force_magnitude2;
  }
  return force;  
}

__device__
float2 get_force_opt4(float3 pos_data, float3 * data_old, int num_particles) {
  // sum force from every other particle based on mass, position of both particles
  float2 force = {0, 0};

  float3 other_data1, other_data2, other_data3, other_data4;
  float x_dist1, y_dist1, dist_cubed1, x_dist2, y_dist2, dist_cubed2;
  float x_dist3, y_dist3, dist_cubed3, x_dist4, y_dist4, dist_cubed4;

  float force_magnitude1, force_magnitude2, force_magnitude3, force_magnitude4;
  float soft_factor = SOFT_FACTOR;
  for (int i = 0; i < num_particles; i+=4)
  {
    other_data1 = data_old[i];    
    other_data2 = data_old[i + 1];
    other_data3 = data_old[i + 2];
    other_data4 = data_old[i + 3];

    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;
    dist_cubed1 = pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + soft_factor, 1.5f); 

    force_magnitude1 = pos_data.z * other_data1.z / dist_cubed1; 
 
    x_dist2 = pos_data.x - other_data2.x;
    y_dist2 = pos_data.y - other_data2.y;
    dist_cubed2 = pow(x_dist2 * x_dist2 + y_dist2 * y_dist2 + soft_factor, 1.5f);    

    force_magnitude2 = pos_data.z * other_data2.z / dist_cubed2; 

    x_dist3 = pos_data.x - other_data3.x;
    y_dist3 = pos_data.y - other_data3.y;
    dist_cubed3 = pow(x_dist3 * x_dist3 + y_dist3 * y_dist3 + soft_factor, 1.5f); 
 
    force_magnitude3 = pos_data.z * other_data3.z / dist_cubed3; 

    x_dist4 = pos_data.x - other_data4.x;
    y_dist4 = pos_data.y - other_data4.y;
    dist_cubed4 = pow(x_dist4 * x_dist4 + y_dist4 * y_dist4 + soft_factor, 1.5f);    

    force_magnitude4 = pos_data.z * other_data4.z / dist_cubed4; 
    force.x += x_dist1 * force_magnitude1 + x_dist2 * force_magnitude2 + 
               x_dist3 * force_magnitude3 + x_dist4 * force_magnitude4;
    force.y += y_dist1 * force_magnitude1 + y_dist2 * force_magnitude2 + 
               y_dist3 * force_magnitude3 + y_dist4 * force_magnitude4;
  }
  return force;  
}

__device__
float2 get_force_opt8(float3 pos_data, float3 * data_old, int num_particles) {
  // sum force from every other particle based on mass, position of both particles
  float2 force = {0, 0};

  float3 other_data1, other_data2, other_data3, other_data4;
  float3 other_data5, other_data6, other_data7, other_data8;

  float x_dist1, y_dist1, dist_cubed1, x_dist2, y_dist2, dist_cubed2;
  float x_dist3, y_dist3, dist_cubed3, x_dist4, y_dist4, dist_cubed4;
  float x_dist5, y_dist5, dist_cubed5, x_dist6, y_dist6, dist_cubed6;
  float x_dist7, y_dist7, dist_cubed7, x_dist8, y_dist8, dist_cubed8;

  float force_magnitude1, force_magnitude2, force_magnitude3, force_magnitude4;
  float force_magnitude5, force_magnitude6, force_magnitude7, force_magnitude8;

  float soft_factor = SOFT_FACTOR;
  for (int i = 0; i < num_particles; i+=8)
  {
    other_data1 = data_old[i];
    other_data2 = data_old[i + 1];
    other_data3 = data_old[i + 2];
    other_data4 = data_old[i + 3];
    other_data5 = data_old[i + 4];
    other_data6 = data_old[i + 5];
    other_data7 = data_old[i + 6];
    other_data8 = data_old[i + 7];

    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;
    dist_cubed1 = pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + soft_factor, 1.5f); 

    force_magnitude1 = pos_data.z * other_data1.z / dist_cubed1; 
 
    x_dist2 = pos_data.x - other_data2.x;
    y_dist2 = pos_data.y - other_data2.y;
    dist_cubed2 = pow(x_dist2 * x_dist2 + y_dist2 * y_dist2 + soft_factor, 1.5f);    

    force_magnitude2 = pos_data.z * other_data2.z / dist_cubed2; 

    x_dist3 = pos_data.x - other_data3.x;
    y_dist3 = pos_data.y - other_data3.y;
    dist_cubed3 = pow(x_dist3 * x_dist3 + y_dist3 * y_dist3 + soft_factor, 1.5f); 
 
    force_magnitude3 = pos_data.z * other_data3.z / dist_cubed3; 

    x_dist4 = pos_data.x - other_data4.x;
    y_dist4 = pos_data.y - other_data4.y;
    dist_cubed4 = pow(x_dist4 * x_dist4 + y_dist4 * y_dist4 + soft_factor, 1.5f);    

    force_magnitude4 = pos_data.z * other_data4.z / dist_cubed4; 

    x_dist5 = pos_data.x - other_data5.x;
    y_dist5 = pos_data.y - other_data5.y;
    dist_cubed5 = pow(x_dist5 * x_dist5 + y_dist5 * y_dist5 + soft_factor, 1.5f); 

    force_magnitude5 = pos_data.z * other_data5.z / dist_cubed5; 
 
    x_dist6 = pos_data.x - other_data6.x;
    y_dist6 = pos_data.y - other_data6.y;
    dist_cubed6 = pow(x_dist6 * x_dist6 + y_dist6 * y_dist6 + soft_factor, 1.5f);    

    force_magnitude6 = pos_data.z * other_data6.z / dist_cubed6; 

    x_dist7 = pos_data.x - other_data7.x;
    y_dist7 = pos_data.y - other_data7.y;
    dist_cubed7 = pow(x_dist7 * x_dist7 + y_dist7 * y_dist7 + soft_factor, 1.5f); 
 
    force_magnitude7 = pos_data.z * other_data7.z / dist_cubed7; 

    x_dist8 = pos_data.x - other_data8.x;
    y_dist8 = pos_data.y - other_data8.y;
    dist_cubed8 = pow(x_dist8 * x_dist8 + y_dist8 * y_dist8 + soft_factor, 1.5f);    

    force_magnitude8 = pos_data.z * other_data8.z / dist_cubed8; 

    force.x += x_dist1 * force_magnitude1 + x_dist2 * force_magnitude2 + 
               x_dist3 * force_magnitude3 + x_dist4 * force_magnitude4 +
               x_dist5 * force_magnitude5 + x_dist6 * force_magnitude6 + 
               x_dist7 * force_magnitude7 + x_dist8 * force_magnitude8;

    force.y += y_dist1 * force_magnitude1 + y_dist2 * force_magnitude2 + 
               y_dist3 * force_magnitude3 + y_dist4 * force_magnitude4 +
               y_dist5 * force_magnitude5 + y_dist6 * force_magnitude6 + 
               y_dist7 * force_magnitude7 + y_dist8 * force_magnitude8;
  }
  return force;  
}

__global__
void simple_kernel(float2 * vels_old, float2 * vels_new, float3 * data_old, float3 * data_new, float dt, int num_particles) {
  // each thread handles a particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while (i < num_particles)
  {
    float2 force = get_force(data_old[i], data_old, num_particles);
    
    vels_new[i].x = vels_old[i].x + force.x * dt / data_old[i].z;
    vels_new[i].y = vels_old[i].y + force.y * dt / data_old[i].z;
    
    data_new[i].x = data_old[i].x + vels_new[i].x * dt; 
    data_new[i].y = data_old[i].y + vels_new[i].y * dt;

    i += blockDim.x * gridDim.x;
  }
}

__global__
void pxp_kernel(float2 * vels_old, float2 * vels_new, float3 * data_old, float3 * data_new, float dt, int num_particles) {
  extern __shared__ float3 sdata[];
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  while (i < num_particles)
  {
    float2 force;
    force.x = 0;
    force.y = 0; 

    float3 pos_data = data_old[i];
    // NOTE: num_particles is a multiple of num_threads_per_block.
    for (int num_tile = 0; num_tile * blockDim.x < num_particles; num_tile++)
    {
      __syncthreads();
      sdata[tid] = data_old[num_tile * blockDim.x + tid];
      __syncthreads();
      float2 block_force = get_force(pos_data, sdata, blockDim.x);
      force.x += block_force.x;
      force.y += block_force.y;
    }    
    
    vels_new[i].x = vels_old[i].x + force.x * dt / data_old[i].z; // TODO: replace data_old[i] with pos_data
    vels_new[i].y = vels_old[i].y + force.y * dt / data_old[i].z;
    
    data_new[i].x = data_old[i].x + vels_new[i].x * dt; 
    data_new[i].y = data_old[i].y + vels_new[i].y * dt;

    i += blockDim.x * gridDim.x;
  }
}

__global__
void pxp_opt_forces_kernel(float2 * forces, float2 * vels_old, float2 * vels_new, float3 * data_old, 
                           float3 * data_new, float dt, int num_particles) 
{
  extern __shared__ float3 sdata[];
  
  int tile_id = blockIdx.x;
  int tid = threadIdx.x;
  
  int num_tiles_per_col = num_particles / blockDim.x;
  int num_tiles = num_particles * num_particles / (blockDim.x * blockDim.x);

  while (tile_id < num_tiles)
  {
    int rid = (tile_id % num_tiles_per_col) * blockDim.x + tid;
    int cid = (tile_id/num_tiles_per_col) * blockDim.x + tid;
    
    sdata[tid] = data_old[cid];
 
    __syncthreads();

    float2 block_force = get_force(data_old[rid], sdata, blockDim.x);
    atomicAdd(&forces[rid].x, block_force.x);
    atomicAdd(&forces[rid].y, block_force.y);
    
    __syncthreads();

    tile_id += gridDim.x;
  }
}

__global__
void pxp_opt_particles_kernel(float2 * forces, float2 * vels_old, float2 * vels_new, float3 * data_old, 
                         float3 * data_new, float dt, int num_particles)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  while (i < num_particles)
  {
    float2 force = forces[i];
    
    vels_new[i].x = vels_old[i].x + force.x * dt / data_old[i].z; // TODO: replace data_old[i] with pos_data
    vels_new[i].y = vels_old[i].y + force.y * dt / data_old[i].z;
    
    data_new[i].x = data_old[i].x + vels_new[i].x * dt; 
    data_new[i].y = data_old[i].y + vels_new[i].y * dt;

    i += blockDim.x * gridDim.x;
  }
}
 
void call_interact_kernel(float dt) {
  // call kernel
  if (algorithm == SIMPLE)
  {
    simple_kernel<<<num_blocks, num_threads_per_block>>>(particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);
  }
  else if (algorithm == PXP)
  {
    pxp_kernel<<<num_blocks, num_threads_per_block, num_threads_per_block * sizeof(float3)>>>
                                                        (particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);
  }
  else if (algorithm == PXP_OPT)
  {
    float2 * forces;
    gpuErrChk(cudaMalloc((void **) &forces, num_particles * sizeof(float2))); 
    gpuErrChk(cudaMemset(forces, 0, num_particles * sizeof(float2)));

    pxp_opt_forces_kernel<<<num_blocks, num_threads_per_block, num_threads_per_block * sizeof(float3)>>>
                                                         (forces, particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);

    pxp_opt_particles_kernel<<<num_blocks, num_threads_per_block>>>(forces, particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);

    gpuErrChk(cudaFree(forces));
  } 
  else {
    std::cout << "Invalid algorithm supplied: " << algorithm << "\n";
  }

  // update pingpong
  pingpong = 1 - pingpong;
}


void get_particle_data(float * h_particle_data, float * h_particle_vels) {
  // copy GPU data into particle_data, particle_vels array
  gpuErrChk(cudaMemcpy(h_particle_data, particle_data[1 - pingpong], sizeof(float) * 3 * num_particles, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(h_particle_vels, particle_vels[1 - pingpong], sizeof(float) * 2 * num_particles, cudaMemcpyDeviceToHost));
}
