#ifndef PARALLEL_SCAN_CUH_
#define PARALLEL_SCAN_CUH_

__global__
void gpu_scan_hillissteele(int* input, int* output, int n)
{
  extern __shared__ float sh_data_[];

  int idx_ = threadIdx.x;

  sh_data_[idx_] = input[idx_];
  __syncthreads();

  for (int offset = 1; offset < n; offset <<= 1)
  {
    if (idx_ >= offset)
      sh_data_[idx_] += sh_data_[idx_ - offset];

    __syncthreads();
  }

  output[idx_] = sh_data_[idx_];
}

#endif
