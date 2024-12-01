//
// (C) 2021, E. Wes Bethel
// Created code harness for the sobel filter
// (C) 2024, Brandon Watanabe
// Modified code to be nbody simulation
//
// Usage:
//

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>  


struct Body
{
   float mass;
   std::vector<float> velocity;
   std::vector<float> position;

   bool operator==(const Body& other) const
   {
      return mass == other.mass && velocity == other.velocity && position == other.position;
   }
};

char output_fname[] = "../data/positions.csv";

#define G 6.67430e-11
#define SOLAR_MASS 1.989e30


std::vector<float> 
compute_forces(std::vector<Body>* bodies, Body i, int dimensions)
{
   std::vector<float> total_force(dimensions, 0.0);

   for(int j = 0; j < bodies->size(); j++)
   {
      if(i == (*bodies)[j])
         continue;

      std::vector<float> dx(dimensions, 0.0);
      float r = 0.0;
      float r_norm = 0.0;

      for (int idx = 0; idx < dimensions; idx++)
      {
         dx[idx] = bodies->at(j).position[bodies->at(j).position.size() - dimensions + idx] - i.position[i.position.size() - dimensions + idx];
         r += dx[idx] * dx[idx];
      }

      r_norm = sqrt(r);
      if (r_norm == 0.0)
         continue;

      for (int idx = 0; idx < dimensions; idx++)
      {
         total_force.push_back((G * i.mass * bodies->at(j).mass * dx[idx]) / (r_norm * r_norm * r_norm));
      }
   }

   return total_force;
}

void 
update_bodies(std::vector<Body>& bodies, std::vector<float>& forces, float dt, int dimensions)
{
   for (int i = 0; i < bodies.size(); i++)
   {
      for (int idx = 0; idx < dimensions; idx++)
      {
         bodies[i].velocity[idx] += forces[i * dimensions + idx] / bodies[i].mass * dt;
         bodies[i].position.push_back(bodies[i].velocity[idx] * dt);
      }
   }
}

void 
do_nBody_calculation(std::vector<Body>* bodies, int dimensions, int timestep, int final_time)
{
   for(int t = 0; t < final_time; t+=timestep)
   {
      std::vector<float> forces(bodies->size() * dimensions, 0.0);
      for(int i = 0; i < bodies->size(); i++)
      {
         std::vector<float> force = compute_forces(bodies, bodies->at(i), dimensions);
         for (int idx = 0; idx < dimensions; idx++)
         {
            forces[i * dimensions + idx] = force[idx];
         }
         std::cout << std::endl;
      }

      update_bodies(*bodies, forces, timestep, dimensions);
   }
}

// This function will initialize the bodies with 
// random masses, initial velocities, and initial positions
// mass is in the range 1.0e-6 to SOLAR_MASS
// velocity is in the range -1.0 to 1.0
// position is in the range -1.0 to 1.0
std::vector<Body>
init_random_bodies(int dimensions, int N)
{
   std::vector<Body> bodies(N);
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<float> mass_dist(1.0e-6, SOLAR_MASS);
   std::uniform_real_distribution<float> velocity_dist(-1.0, 1.0);
   std::uniform_real_distribution<float> position_dist(-1.0, 1.0);

   for(int i = 0; i < N; i++)
   {
      Body body;
      body.mass = mass_dist(gen);
      body.velocity.resize(dimensions);
      body.position.resize(dimensions); // Resize the position vector

      for (int j = 0; j < dimensions; j++)
      {
         body.velocity[j] = velocity_dist(gen);
         body.position[j] = position_dist(gen);
      }
      bodies[i] = body;
   }

   return bodies;
}


void 
save_positions(const std::vector<Body>& bodies) 
{
   FILE *fp = fopen(output_fname, "w");
   if (fp == NULL)
   {
      std::cerr << "Error: could not open file " << output_fname << " for writing" << std::endl;
      exit(1);
   }

   // Print header
   fprintf(fp, "body_num,x,y,z\n");

   for (int i = 0; i < bodies.size(); i++)
   {
      const std::vector<float>& pos = bodies[i].position;
      if (pos.size() % 3 != 0) {
         std::cerr << "Error: body " << i << " does not have the correct number of position coordinates" << std::endl;
         fclose(fp);
         exit(1);
      }

      for (int j = 0; j < pos.size(); j += 3) {
         fprintf(fp, "%d, %f, %f, %f\n", i, pos[j], pos[j+1], pos[j+2]);
      }
   }

   fclose(fp);
}

int
main (int ac, char *av[])
{
   #pragma omp parallel
   {
      // ID of the thread in the current team
      int thread_id = omp_get_thread_num();
      // Number of threads in the current team
      int nthreads = omp_get_num_threads();

      #pragma omp critical
      {
         std::cout << "Hello world, I'm thread " << thread_id << " out of " << nthreads << " total threads. " << std::endl; 
      }
   }
   int dimensions = 3;
   int N = 15;
   int timestep = 60*60;
   int final_time = timestep * 24;
   std::vector<Body> bodies = init_random_bodies(dimensions, N);
   

   // do the processing =======================
   std::cout << "Starting nbody calculation" << std::endl;

   std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

   do_nBody_calculation(&bodies, dimensions, timestep, final_time);

   std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> elapsed = end_time - start_time;
   std::cout << " Elapsed time is : " << elapsed.count() << " " << std::endl;

   save_positions(bodies);

   
   
}

// eof
