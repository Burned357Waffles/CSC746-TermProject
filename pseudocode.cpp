__global__ void 
compute_forces()
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= N) return;

   double total_force[DIM] = {0.0, 0.0, 0.0};

   for(int j = 0; j < N; j++)
   {
      if(i == j)
         continue;

      double dx[DIM] = {0.0, 0.0, 0.0};
      double r = 0.0;
      double r_norm = 0.0;

      for (int idx = 0; idx < DIM; idx++)
      {
         dx[idx] = bodies[j].position[idx] - bodies[i].position[idx];
         r += dx[idx] * dx[idx];
      }

      r_norm = sqrt(r);
      if (r_norm == 0.0)
         continue;

      for (int idx = 0; idx < DIM; idx++)
      {
         double f = (G * bodies[i].mass * bodies[j].mass * dx[idx]) / (r_norm * r_norm * r_norm);
         total_force[idx] += f;
      }
   }

   for (int idx = 0; idx < DIM; idx++)
   {
      forces[i * DIM + idx] = total_force[idx];
   }
}

__global__ void 
update_bodies()
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= N) return;

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

void 
do_nBody_calculation()
{
   double* forces;
   gpuErrchk(cudaMallocManaged(&forces, N * DIM * sizeof(double)));
   
   for(int t = 0; t < final_time; t+=timestep)
   {
      memset(forces, 0, N * DIM * sizeof(double));

      compute_forces<<<num_blocks, threads_per_block>>>(bodies, forces, N);
      gpuErrchk(cudaDeviceSynchronize());

      update_bodies<<<num_blocks, threads_per_block>>>(bodies, forces, timestep, N, record_histories, history_index, velocity_history, position_history);
      gpuErrchk(cudaDeviceSynchronize());
   }

   gpuErrchk(cudaFree(forces));
}