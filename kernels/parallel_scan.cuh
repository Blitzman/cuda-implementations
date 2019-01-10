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

__global__
void gpu_scan_blelloch(int* input, int* output, int n)
{
  extern __shared__ float sh_data_[];

  int idx_ = 2 * blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_ < n)
    sh_data_[threadIdx.x] = input[idx_];

  __syncthreads();

  for (unsigned int offset = 1; offset <= n; offset <<=1)
  {
    int idx_s_ = (threadIdx.x + 1) * 2 * offset - 1;
    if (idx_s_ < n)
      sh_data_[idx_s_] += sh_data_[idx_s_ - offset];

    __syncthreads();
  }

  for (unsigned int offset = n/4; offset > 0; offset >>=1)
  {
    int idx_s_ = (threadIdx.x + 1) * 2 * offset - 1;
    if (idx_s_ + offset < n)
      sh_data_[idx_s_ + offset] += sh_data_[idx_s_];

    __syncthreads();
  }

  if (idx_ < n)
    output[idx_] = sh_data_[threadIdx.x];
}

#endif
