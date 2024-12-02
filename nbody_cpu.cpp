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
   float velocity[3];
   float position[3];
   std::vector<float> velocity_history;
   std::vector<float> position_history;

   bool operator==(const Body& other) const
   {
      return mass == other.mass && 
             std::equal(std::begin(velocity), std::end(velocity), std::begin(other.velocity)) &&
             std::equal(std::begin(position), std::end(position), std::begin(other.position));
   }
};

char output_fname[] = "../data/positions.csv";

#define G 6.67430e-11
#define SOLAR_MASS 1.989e30
#define MERCURY_MASS 3.285e23
#define VENUS_MASS 4.867e24
#define EARTH_MASS 5.972e24
#define MARS_MASS 6.390e23
#define JUPITER_MASS 1.898e27
#define SATURN_MASS 5.683e26
#define URANUS_MASS 8.681e25
#define NEPTUNE_MASS 1.024e26
#define ASTEROID_MASS 1.0e12
#define AU 1.496e11


std::vector<float> 
compute_forces(std::vector<Body>& bodies, Body i, int N, int dimensions)
{
   std::vector<float> total_force(dimensions, 0.0);

   for(int j = 0; j < N; j++)
   {
      if(i == bodies[j])
         continue;

      std::vector<float> dx(dimensions, 0.0);
      float r = 0.0;
      float r_norm = 0.0;

      for (int idx = 0; idx < dimensions; idx++)
      {
         dx[idx] = bodies[j].position[idx] - i.position[idx];
         r += dx[idx] * dx[idx];
      }

      r_norm = sqrt(r);
      if (r_norm == 0.0)
         continue;

      for (int idx = 0; idx < dimensions; idx++)
      {
         float f = (G * i.mass * bodies[j].mass * dx[idx]) / (r_norm * r_norm * r_norm);
         total_force[idx] += f;
      }
   }

   return total_force;
}

void 
update_bodies(std::vector<Body>& bodies, std::vector<float>& forces, float dt, int N, int dimensions)
{
   for (int i = 0; i < N; i++)
   {
      for (int idx = 0; idx < dimensions; idx++)
      {
         bodies[i].velocity[idx] += forces[i * dimensions + idx] / bodies[i].mass * dt;
         bodies[i].position[idx] += bodies[i].velocity[idx] * dt;

         bodies[i].velocity_history.push_back(bodies[i].velocity[idx]);
         bodies[i].position_history.push_back(bodies[i].position[idx]);
      }
   }
}

void 
do_nBody_calculation(std::vector<Body>& bodies, int N, int dimensions, int timestep, unsigned int final_time)
{
   for(int t = 0; t < final_time; t+=timestep)
   {
      std::vector<float> forces(N * dimensions, 0.0);

      #pragma omp parallel for
      for(int i = 0; i < N; i++)
      {
         std::vector<float> force = compute_forces(bodies, bodies[i], N, dimensions);
         for (int idx = 0; idx < dimensions; idx++)
         {
            forces[i * dimensions + idx] = force[idx];
         }
      }

      update_bodies(bodies, forces, timestep, N, dimensions);
   }
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

std::vector<Body>
init_random_bodies(int dimensions, int N)
{
   std::vector<Body> bodies(N);
   std::random_device rd;
   std::mt19937 gen(rd());
   //std::uniform_real_distribution<float> mass_dist(ASTEROID_MASS, ASTEROID_MASS);
   std::uniform_real_distribution<float> mass_dist(ASTEROID_MASS, JUPITER_MASS);
   std::uniform_real_distribution<float> velocity_dist(-50.0e3, 50.0e3); // Velocity in m/s
   std::uniform_real_distribution<float> position_dist(-AU, AU);

   for(int i = 0; i < N; i++)
   {
      Body body;
      body.mass = mass_dist(gen);
      //body.velocity.resize(dimensions);
      //body.position.resize(dimensions); // Resize the position vector

      for (int j = 0; j < dimensions; j++)
      {
         body.velocity[j] = velocity_dist(gen);
         body.position[j] = position_dist(gen);
      }
      bodies[i] = body;
   }

   // print all the bodies
   /*
   for (int i = 0; i < N; i++)
   {
      std::cout << "Body " << i << " mass: " << bodies[i].mass << std::endl;
      std::cout << "Body " << i << " velocity: " << bodies[i].velocity[0] << ", " << bodies[i].velocity[1] << ", " << bodies[i].velocity[2] << std::endl;
      std::cout << "Body " << i << " position: " << bodies[i].position[0] / AU << ", " << bodies[i].position[1] / AU  << ", " << bodies[i].position[2] / AU  << std::endl;
   }
   */

   return bodies;
}

std::vector<Body>
init_solar_system(int dimensions)
{
   std::vector<Body> bodies(9);
   bodies[0].mass = SOLAR_MASS;
   bodies[0].velocity[0] = 0;
   bodies[0].velocity[1] = 0;
   bodies[0].velocity[2] = 0;
   bodies[0].position[0] = 0;
   bodies[0].position[1] = 0;
   bodies[0].position[2] = 0;
 
   bodies[1].mass = MERCURY_MASS;
   bodies[1].velocity[0] = 0;
   bodies[1].velocity[1] = 47.87e3;
   bodies[1].velocity[2] = 0;
   bodies[1].position[0] = 0.39 * AU;
   bodies[1].position[1] = 0;
   bodies[1].position[2] = 0;
   
   bodies[2].mass = VENUS_MASS;
   bodies[2].velocity[0] = 0;
   bodies[2].velocity[1] = 35.02e3;
   bodies[2].velocity[2] = 0;
   bodies[2].position[0] = 0.72 * AU;
   bodies[2].position[1] = 0;
   bodies[2].position[2] = 0;

   bodies[3].mass = EARTH_MASS;
   bodies[3].velocity[0] = 0;
   bodies[3].velocity[1] = 29.78e3;
   bodies[3].velocity[2] = 0;
   bodies[3].position[0] = AU;
   bodies[3].position[1] = 0;
   bodies[3].position[2] = 0;

   bodies[4].mass = MARS_MASS;
   bodies[4].velocity[0] = 0;
   bodies[4].velocity[1] = 24.07e3;
   bodies[4].velocity[2] = 0;
   bodies[4].position[0] = 1.52 * AU;
   bodies[4].position[1] = 0;
   bodies[4].position[2] = 0;

   bodies[5].mass = JUPITER_MASS;
   bodies[5].velocity[0] = 0;
   bodies[5].velocity[1] = 13.07e3;
   bodies[5].velocity[2] = 0;
   bodies[5].position[0] = 5.20 * AU;
   bodies[5].position[1] = 0;
   bodies[5].position[2] = 0;

   bodies[6].mass = SATURN_MASS;
   bodies[6].velocity[0] = 0;
   bodies[6].velocity[1] = 9.69e3;
   bodies[6].velocity[2] = 0;
   bodies[6].position[0] = 9.58 * AU;
   bodies[6].position[1] = 0;
   bodies[6].position[2] = 0;

   bodies[7].mass = URANUS_MASS;
   bodies[7].velocity[0] = 0;
   bodies[7].velocity[1] = 6.81e3;
   bodies[7].velocity[2] = 0;
   bodies[7].position[0] = 19.22 * AU;
   bodies[7].position[1] = 0;
   bodies[7].position[2] = 0;

   bodies[8].mass = NEPTUNE_MASS;
   bodies[8].velocity[0] = 0;
   bodies[8].velocity[1] = 5.43e3;
   bodies[8].velocity[2] = 0;
   bodies[8].position[0] = 30.05 * AU;
   bodies[8].position[1] = 0;
   bodies[8].position[2] = 0;

   return bodies;
}


void 
write_data_to_file(const std::vector<Body>& bodies, int N) 
{
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
      const std::vector<float>& pos = bodies[i].position_history;
      const std::vector<float>& vel = bodies[i].velocity_history;
      if (pos.size() % 3 != 0) {
         std::cerr << "Error: body " << i << " does not have the correct number of position coordinates" << std::endl;
         fclose(fp);
         exit(1);
      }

      if (vel.size() % 3 != 0) {
         std::cerr << "Error: body " << i << " does not have the correct number of veelocities" << std::endl;
         fclose(fp);
         exit(1);
      }


      for (int j = 0; j < pos.size(); j += 3) {
         fprintf(fp, "%d,%f,%f,%f,%f,%f,%f,%f\n", i, bodies[i].mass, vel[j], vel[j+1], vel[j+2], pos[j], pos[j+1], pos[j+2]);
      }
   }

   fclose(fp);

   std::cout << "Data written to " << output_fname << std::endl;
}

int
main (int ac, char *av[])
{
   if (ac < 2) {
      std::cerr << "Usage: " << av[0] << " <number_of_bodies>" << std::endl;
      return 1;
   }

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

   int N = std::stoi(av[1]);

   int dimensions = 3;
   int timestep = 60 * 60;
   unsigned int final_time = timestep * 24 * 365; 
   std::vector<Body> bodies;

   if (N == -1)
   {
      N = 9;
      bodies = init_solar_system(dimensions);
   }
   else 
   {
      bodies = init_random_bodies(dimensions, N);
   }
   
   std::cout << "Number of bodies: " << N << std::endl;

   // do the processing =======================
   std::cout << "Starting nbody calculation" << std::endl;

   std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

   do_nBody_calculation(bodies, N, dimensions, timestep, final_time);

   std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> elapsed = end_time - start_time;
   std::cout << " Elapsed time is : " << elapsed.count() << " " << std::endl;

   write_data_to_file(bodies, N);
}

// eof