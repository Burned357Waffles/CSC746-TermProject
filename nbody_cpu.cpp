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

   bool operator==(const Body& other) const
   {
      return mass == other.mass && 
             std::equal(std::begin(velocity), std::end(velocity), std::begin(other.velocity)) &&
             std::equal(std::begin(position), std::end(position), std::begin(other.position));
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


void 
compute_forces(Body* bodies, Body i, int N, double* total_force)
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

void 
update_bodies(Body* bodies, const double* forces, const double dt, const int N, const bool record_histories, const int history_index, double** velocity_history, double** position_history)
{
   //#pragma omp parallel for
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
            velocity_history[history_index * N + i][idx] = bodies[i].velocity[idx];
            position_history[history_index * N + i][idx] = bodies[i].position[idx];
         }
      }
    
   }
}

void 
do_nBody_calculation(Body* bodies, const int N, const int timestep, const unsigned long long final_time, const bool record_histories, double** velocity_history, double** position_history)
{
   int history_index = 1;
   double* forces = new double[N * DIM];
   
   for(int t = 0; t < final_time; t+=timestep)
   {
      memset(forces, 0, N * DIM * sizeof(double));

      #pragma omp parallel 
      {
         #ifdef LIKWID_PERFMON
         LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
         #endif

         #pragma omp for
         for(int i = 0; i < N; i++)
         {
            double total_force[DIM] = {0.0, 0.0, 0.0};            
            compute_forces(bodies, bodies[i], N, total_force);
            for (int idx = 0; idx < DIM; idx++)
            {
               forces[i * DIM + idx] = total_force[idx];
            }
         }

         #ifdef LIKWID_PERFMON
         LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
         #endif
      }

      update_bodies(bodies, forces, timestep, N, record_histories, history_index, velocity_history, position_history);
      history_index++;
   }

   delete[] forces;
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
   Body* bodies = new Body[N];
   std::random_device rd;
   std::mt19937 gen(rd());
   //std::uniform_real_distribution<double> mass_dist(ASTEROID_MASS, ASTEROID_MASS);
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

   
   bodies[0].mass = SOLAR_MASS;  
   bodies[0].velocity[0] = 0;
   bodies[0].velocity[1] = 0;
   bodies[0].velocity[2] = 0;
   bodies[0].position[0] = 0;
   bodies[0].position[1] = 0;
   bodies[0].position[2] = 0;

   bodies[1].mass = ASTEROID_MASS;
   bodies[1].velocity[0] = 22365.5;
   bodies[1].velocity[1] = 24955.3;
   bodies[1].velocity[2] = 28634.1;
   bodies[1].position[0] = -1.77419e+09;
   bodies[1].position[1] = 1.52822e+10;
   bodies[1].position[2] = -2.62286e+10;
   

   return bodies;
}


Body*
init_solar_system()
{
   Body* bodies = new Body[9];

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
write_data_to_file(Body* bodies, const int N, const int timestep, const unsigned long long final_time, double** velocity_history, double** position_history) 
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
         const double* pos = position_history[j * N + i];
         const double* vel = velocity_history[j * N + i];
         fprintf(fp, "%d,%f,%f,%f,%f,%f,%f,%f\n", i, bodies[i].mass, vel[0], vel[1], vel[2], pos[0], pos[1], pos[2]);
      }
   }

   fclose(fp);

   std::cout << "Data written to " << output_fname << std::endl;
}

double** allocate_history(int N, int history_length)
{
   double** history = new double*[history_length * N];
   for (int i = 0; i < history_length * N; i++)
   {
      history[i] = new double[DIM];
   }
   return history;
}

void free_history(double** history, int N, int history_length)
{
   for (int i = 0; i < history_length * N; i++)
   {
      delete[] history[i];
   }
   delete[] history;
}

int
main (int ac, char *av[])
{
   if (ac < 5) {
      std::cerr << "Usage: " << av[0] << " <number_of_bodies> <record_histories> <timestep_modifier> <final_time_modifier>" << std::endl;
      return 1;
   }

   LIKWID_MARKER_INIT;

   #pragma omp parallel
   {
      // ID of the thread in the current team
      int thread_id = omp_get_thread_num();
      // Number of threads in the current team
      int nthreads = omp_get_num_threads();

      #pragma omp critical
      {
         std::cout << "Hello world, I'm thread " << thread_id << " out of " << nthreads << " total threads. " << std::endl; 
         // Each thread must add itself to the Marker API, therefore must be
         // in parallel region
         LIKWID_MARKER_THREADINIT;
         // Register region name
         LIKWID_MARKER_REGISTER(MY_MARKER_REGION_NAME);
      }
   }

   int N = std::stoi(av[1]);
   bool record_histories = std::stoi(av[2]);
   int timestep_modifier = std::stoi(av[3]);
   int final_time_modifier = std::stoi(av[4]);

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
   double** velocity_history = allocate_history(N, history_length);
   double** position_history = allocate_history(N, history_length);

   if (record_histories)
   {
      for (int i = 0; i < N; i++)
      {
         for (int idx = 0; idx < DIM; idx++)
         {
            velocity_history[i][idx] = bodies[i].velocity[idx];
            position_history[i][idx] = bodies[i].position[idx];
         }
      }
   }

   //velocity_history.resize(bodies.size() * DIM);
   //position_history.resize(bodies.size() * DIM);
   
   std::cout << "Number of bodies: " << N << std::endl;

   // do the processing =======================
   std::cout << "Starting nbody calculation" << std::endl;

   std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

   do_nBody_calculation(bodies, N, timestep, final_time, record_histories, velocity_history, position_history);

   std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> elapsed = end_time - start_time;
   std::cout << " Elapsed time is : " << elapsed.count() << " " << std::endl;

   if (record_histories)
      write_data_to_file(bodies, N, timestep, final_time, velocity_history, position_history);
   else
      std::cout << "Histories were not recorded" << std::endl;

   free_history(velocity_history, N, history_length);
   free_history(position_history, N, history_length);
   delete[] bodies;

   // Close Marker API and write results to file for further evaluation done
   // by likwid-perfctr
   LIKWID_MARKER_CLOSE;

   return 0;

}

// eof