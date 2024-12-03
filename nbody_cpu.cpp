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

#define DIM 3
#define HOUR 3600
#define EARTH_DAY 3600 * 24
#define EARTH_YEAR 3600 * 24 * 365

struct Body
{
   double mass;
   double velocity[DIM];
   double position[DIM];
   std::vector<double> velocity_history;
   std::vector<double> position_history;

   bool operator==(const Body& other) const
   {
      return mass == other.mass && 
             std::equal(std::begin(velocity), std::end(velocity), std::begin(other.velocity)) &&
             std::equal(std::begin(position), std::end(position), std::begin(other.position));
   }
};

char output_fname[] = "../data/positions.csv";

const double G = 6.67430e-11;
const double SOLAR_MASS = 1.989e30;
const double MERCURY_MASS = 3.285e23;
const double VENUS_MASS = 4.867e24;
const double EARTH_MASS = 5.972e24;
const double MARS_MASS = 6.39e23;
const double JUPITER_MASS = 1.898e27;
const double SATURN_MASS = 5.683e26;
const double AU = 1.496e11;
const double URANUS_MASS = 8.681e25;
const double NEPTUNE_MASS = 1.024e26;
const double ASTEROID_MASS = 1.0e12;

std::vector<std::vector<double>> velocity_history;
std::vector<std::vector<double>> position_history;


std::vector<double> 
compute_forces(std::vector<Body>& bodies, Body i, int N)
{
   std::vector<double> total_force(DIM, 0.0);

   for(int j = 0; j < N; j++)
   {
      if(i == bodies[j])
         continue;

      std::vector<double> dx(DIM, 0.0);
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

   return total_force;
}

void 
update_bodies(std::vector<Body>& bodies, const std::vector<double>& forces, const double dt, const int N, const bool record_histories)
{
   //#pragma omp parallel for
   for (int i = 0; i < N; i++)
   {
      for (int idx = 0; idx < DIM; idx++)
      {
         bodies[i].velocity[idx] += forces[i * DIM + idx] / bodies[i].mass * dt;
         bodies[i].position[idx] += bodies[i].velocity[idx] * dt;

         //bodies[i].velocity_history.push_back(bodies[i].velocity[idx]);
         //bodies[i].position_history.push_back(bodies[i].position[idx]);
      }

      if (record_histories)
      {
         velocity_history[i].insert(velocity_history[i].end(), std::begin(bodies[i].velocity), std::end(bodies[i].velocity));
         position_history[i].insert(position_history[i].end(), std::begin(bodies[i].position), std::end(bodies[i].position));
      }
    
   }
}

void 
do_nBody_calculation(std::vector<Body>& bodies, const int N, const int timestep, const unsigned long long final_time, const bool record_histories)
{
   for(int t = 0; t < final_time; t+=timestep)
   {
      std::vector<double> forces(N * DIM, 0.0);

      #pragma omp parallel for
      for(int i = 0; i < N; i++)
      {
         std::vector<double> force = compute_forces(bodies, bodies[i], N);
         for (int idx = 0; idx < DIM; idx++)
         {
            forces[i * DIM + idx] = force[idx];
         }
      }

      update_bodies(bodies, forces, timestep, N, record_histories);
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
init_random_bodies(const int N)
{
   std::vector<Body> bodies(N);
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
      //body.velocity.resize(DIM);
      //body.position.resize(DIM); // Resize the position vector

      for (int j = 0; j < DIM; j++)
      {
         body.velocity[j] = velocity_dist(gen);
         body.position[j] = position_dist(gen);
      }
      bodies[i] = body;
   }

   /*
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
   */

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
init_solar_system()
{
   std::vector<Body> bodies(9);

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
write_data_to_file(const std::vector<Body>& bodies, const int N) 
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
      const std::vector<double>& pos = position_history[i];
      const std::vector<double>& vel = velocity_history[i];
      if (pos.size() % 3 != 0) {
         std::cerr << "Error: body " << i << " does not have the correct number of position coordinates" << std::endl;
         fclose(fp);
         exit(1);
      }

      if (vel.size() % 3 != 0) {
         std::cerr << "Error: body " << i << " does not have the correct number of velocities" << std::endl;
         fclose(fp);
         exit(1);
      }

      for (size_t j = 0; j < pos.size(); j += 3) {
         fprintf(fp, "%d,%f,%f,%f,%f,%f,%f,%f\n", i, bodies[i].mass, vel[j], vel[j+1], vel[j+2], pos[j], pos[j+1], pos[j+2]);
      }
   }

   fclose(fp);

   std::cout << "Data written to " << output_fname << std::endl;
}

int
main (int ac, char *av[])
{
   if (ac < 5) {
      std::cerr << "Usage: " << av[0] << " <number_of_bodies> <record_histories> <timestep_modifier> <final_time_modifier>" << std::endl;
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
   bool record_histories = std::stoi(av[2]);
   int timestep_modifier = std::stoi(av[3]);
   int final_time_modifier = std::stoi(av[4]);

   int timestep = EARTH_DAY * timestep_modifier;
   unsigned long long final_time = static_cast<unsigned long long>(EARTH_YEAR) * final_time_modifier; 

   std::vector<Body> bodies;
   if (N == -1)
   {
      N = 9;
      bodies = init_solar_system();
   }
   else 
   {
      bodies = init_random_bodies(N);
   }

   velocity_history.resize(bodies.size() * DIM);
   position_history.resize(bodies.size() * DIM);
   
   std::cout << "Number of bodies: " << N << std::endl;

   // do the processing =======================
   std::cout << "Starting nbody calculation" << std::endl;

   std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

   do_nBody_calculation(bodies, N, timestep, final_time, record_histories);

   std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

   std::chrono::duration<double> elapsed = end_time - start_time;
   std::cout << " Elapsed time is : " << elapsed.count() << " " << std::endl;

   if (record_histories)
      write_data_to_file(bodies, N);
   else
      std::cout << "Histories were not recorded" << std::endl;
}

// eof