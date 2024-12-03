//
// (C) 2021, E. Wes Bethel
// Created code harness for the sobel filter
// (C) 2024, Brandon Watanabe
// Modified code to be nbody simulation
//
// Usage:
//

#include <iostream>
#include <chrono>
#include <random>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>  
#include "likwid-stuff.h"

#define DIM 3
#define HOUR 3600
#define EARTH_DAY 3600 * 24
#define EARTH_YEAR 3600 * 24 * 365

typedef double vect_t[DIM];

struct Body
{
   double mass;
   vect_t velocity;
   vect_t position;
   
   __device__ bool operator==(const Body& other) const
   {
      if (mass != other.mass)
         return false;

      for (int i = 0; i < DIM; ++i)
      {
         if (velocity[i] != other.velocity[i] || position[i] != other.position[i])
            return false;
      }

      return true;
   }
};

char output_fname[] = "../data/positions.csv";

const double G = 6.67430e-11;
const double AU = 1.496e11;
const double SOLAR_MASS = 1.989e30;
const double MERCURY_MASS = 3.285e23;
const double VENUS_MASS = 4.867e24;
const double EARTH_MASS = 5.972e24;
const double MARS_MASS = 6.39e23;
const double JUPITER_MASS = 1.898e27;
const double SATURN_MASS = 5.683e26;
const double URANUS_MASS = 8.681e25;
const double NEPTUNE_MASS = 1.024e26;
const double ASTEROID_MASS = 1.0e12;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void 
compute_forces(Body* bodies, Body i, double* total_force, int N)
{
   for(int j = 0; j < N; j++)
   {
      if(i == bodies[j])
         continue;

      double dx[DIM] = {0.0, 0.0, 0.0};
      double r = 0.0;
      double r_norm = 0.0;

      for (int idx = 0; idx < DIM; idx++)
      {
         dx[idx] = bodies[j].position[idx] - i.position[idx];
         r += dx[idx] * dx[idx];
      }

      r_norm = sqrt(r);
      if (r_norm == 0.0)
         continue;

      for (int idx = 0; idx < DIM; idx++)
      {
         double f = (G * i.mass * bodies[j].mass * dx[idx]) / (r_norm * r_norm * r_norm);
         total_force[idx] += f;
      }
   }
}

__device__ void 
update_bodies(Body* bodies, const double* forces, const double dt, const int N, const bool record_histories, const int history_index, double* velocity_history, double* position_history)
{
   for (int i = 0; i < N; i++)
   {
      for (int idx = 0; idx < DIM; idx++)
      {
         bodies[i].velocity[idx] += forces[i * DIM + idx] / bodies[i].mass * dt;
         bodies[i].position[idx] += bodies[i].velocity[idx] * dt;
      }

      if (record_histories)
      {
         for (int idx = 0; idx < DIM; idx++)
         {
            velocity_history[history_index * N * DIM + i * DIM + idx] = bodies[i].velocity[idx];
            position_history[history_index * N * DIM + i * DIM + idx] = bodies[i].position[idx];
         }
      }
   }
}

__global__ void 
do_nBody_calculation(Body* bodies, const int N, const int timestep, const unsigned long long final_time, const bool record_histories, double* velocity_history, double* position_history, double* forces)
{
   extern __shared__ double shared_forces[];

   int history_index = 1;

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;   

   for(int t = 0; t < final_time; t+=timestep)
   {
      for (int i = threadIdx.x; i < N * DIM; i += blockDim.x)
      {
         shared_forces[i] = 0.0;
      }
      __syncthreads();

      //memset(forces, 0, N * DIM * sizeof(double));
      for (int i = index; i < N; i += stride)
      { 
         compute_forces(bodies, bodies[i], shared_forces + i * DIM, N);
      }
      __syncthreads();

      for (int i = index; i < N; i += stride)
      {
         update_bodies(bodies, shared_forces, timestep, N, record_histories, history_index, velocity_history, position_history);
      }
      __syncthreads();
      history_index++;
   }
}

void launch_nBody_calculation(Body* bodies, const int N, const int timestep, const unsigned long long final_time, const bool record_histories, double* velocity_history, double* position_history)
{
   double* forces;
   gpuErrchkcudaMallocManaged(&forces, N * DIM * sizeof(double));

   int blockSize = 256;
   int numBlocks = (N + blockSize - 1) / blockSize;
   do_nBody_calculation<<<numBlocks, blockSize>>>(bodies, N, timestep, final_time, record_histories, velocity_history, position_history, forces);
   gpuErrchk(cudaGetLastError());
   gpuErrchk(cudaDeviceSynchronize());

   cudaFree(forces);
}

// This function will initialize the bodies with 
// random masses, initial velocities, and initial positions
// mass is in the range 1.0e-6 to SOLAR_MASS
// velocity is in the range -1.0 to 1.0
// position is in the range -1.0 to 1.0
// 
// To get an orbit:
// 1. Set the mass of the first body to SOLAR_MASS
// 2. Set the velocity of the first body to 0
// 3. Set the position of the first body to 0
// 4. Body 1 mass: 1e+12
// 5. Body 1 velocity: 22365.5, 24955.3, 28634.1
// 6/ Body 1 position: -1.77419e+09, 1.52822e+10, -2.62286e+10

Body*
init_random_bodies(const int N)
{
   Body* bodies;
   gpuErrchk(cudaMallocManaged(&bodies, N * sizeof(Body)));
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<double> mass_dist(ASTEROID_MASS, SOLAR_MASS);
   std::uniform_real_distribution<double> velocity_dist(-50.0e3, 50.0e3); // Velocity in m/s
   std::uniform_real_distribution<double> position_dist(-AU, AU);

   for(int i = 0; i < N; i++)
   {
      Body body;
      body.mass = mass_dist(gen);

      for (int j = 0; j < DIM; j++)
      {
         body.velocity[j] = velocity_dist(gen);
         body.position[j] = position_dist(gen);
      }
      bodies[i] = body;
   }
   
   return bodies;
}


Body*
init_solar_system()
{
   Body* bodies;
   gpuErrchk(cudaMallocManaged(&bodies, 9 * sizeof(Body)));

   double masses[] = {SOLAR_MASS, MERCURY_MASS, VENUS_MASS, EARTH_MASS, MARS_MASS, JUPITER_MASS, SATURN_MASS, URANUS_MASS, NEPTUNE_MASS};
   double velocities[][3] = {
         {0, 0, 0},
         {0, 47.87e3, 0},
         {0, 35.02e3, 0},
         {0, 29.78e3, 0},
         {0, 24.07e3, 0},
         {0, 13.07e3, 0},
         {0, 9.69e3, 0},
         {0, 6.81e3, 0},
         {0, 5.43e3, 0}
   };
   double positions[][3] = {
         {0, 0, 0},
         {0.39 * AU, 0, 0},
         {0.72 * AU, 0, 0},
         {AU, 0, 0},
         {1.52 * AU, 0, 0},
         {5.20 * AU, 0, 0},
         {9.58 * AU, 0, 0},
         {19.22 * AU, 0, 0},
         {30.05 * AU, 0, 0}
   };

   for (int i = 0; i < 9; ++i) {
      bodies[i].mass = masses[i];
      for (int j = 0; j < 3; ++j) {
         bodies[i].velocity[j] = velocities[i][j];
         bodies[i].position[j] = positions[i][j];
      }
   }

   return bodies;
}

void 
write_data_to_file(Body* bodies, const int N, const int timestep, const unsigned long long final_time, double* velocity_history, double* position_history) 
{
   int num_data_points = (final_time / timestep) + 1;

   FILE *fp = fopen(output_fname, "w");
   if (fp == NULL)
   {
      std::cerr << "Error: could not open file " << output_fname << " for writing" << std::endl;
      exit(1);
   }

   // Print header
   fprintf(fp, "body_num,m,vx,vy,vz,x,y,z\n");

   for (int i = 0; i < N; i++)
   {
      for (int j = 0; j < num_data_points; j++) {
         const double* pos = &position_history[(j * N + i) * DIM];
         const double* vel = &velocity_history[(j * N + i) * DIM];
         fprintf(fp, "%d,%f,%f,%f,%f,%f,%f,%f\n", i, bodies[i].mass, vel[0], vel[1], vel[2], pos[0], pos[1], pos[2]);
      }
   }

   fclose(fp);

   std::cout << "Data written to " << output_fname << std::endl;
}

double* allocate_history(int N, int history_length)
{
   double* history;
   gpuErrchk(cudaMallocManaged(&history, history_length * N * DIM * sizeof(double)));
   return history;
}

void free_history(double* history)
{
   gpuErrchk(cudaFree(history));
}

int
main (int ac, char *av[])
{
   if (ac < 7) {
      std::cerr << "Usage: " << av[0] << " <number_of_bodies> <record_histories> <timestep_modifier> <final_time_modifier> <threads_per_block> <num_blocks>" << std::endl;
      return 1;
   }

   LIKWID_MARKER_INIT;

   int N = std::stoi(av[1]);
   bool record_histories = std::stoi(av[2]);
   int timestep_modifier = std::stoi(av[3]);
   int final_time_modifier = std::stoi(av[4]);
   int threads_per_block = std::stoi(av[5]);
   int num_blocks = std::stoi(av[6]);

   int timestep = HOUR * timestep_modifier;
   unsigned long long final_time = static_cast<unsigned long long>(EARTH_YEAR) * final_time_modifier; 

   Body* bodies = nullptr;
   if (N == -1)
   {
      N = 9;
      bodies = init_solar_system();
   }
   else 
   {
      bodies = init_random_bodies(N);
   }

   int history_length = (final_time / timestep + 1);
   double* velocity_history = allocate_history(N, history_length);
   double* position_history = allocate_history(N, history_length);

   if (record_histories)
   {
      for (int i = 0; i < N; i++)
      {
         for (int idx = 0; idx < DIM; idx++)
         {
            velocity_history[i * DIM + idx] = bodies[i].velocity[idx];
            position_history[i * DIM + idx] = bodies[i].position[idx];
         }
      }
   }

   std::cout << "Number of bodies: " << N << std::endl;

   // do the processing =======================
   std::cout << "Starting nbody calculation" << std::endl;

   std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

   launch_nBody_calculation(bodies, N, timestep, final_time, record_histories, velocity_history, position_history);

   std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> elapsed = end_time - start_time;
   std::cout << " Elapsed time is : " << elapsed.count() << " " << std::endl;

   if (record_histories)
      write_data_to_file(bodies, N, timestep, final_time, velocity_history, position_history);
   else
      std::cout << "Histories were not recorded" << std::endl;

   free_history(velocity_history);
   free_history(position_history);
   gpuErrchk(cudaFree(bodies));

   LIKWID_MARKER_CLOSE;

   return 0;
}

// eof