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

__global__
void gpu_scan_blellochv2(int* input, int* output, int n)
{
    extern __shared__ float sh_data_[];

    int idx_ = threadIdx.x;
    int offset_ = 1;

    sh_data_[idx_] = input[idx_];
    __syncthreads();

    for (unsigned int d = n >> 1; d > 0; d >>= 1)
    {
        if (idx_ < d)
        {
            int idx_l_ = offset_ * (2 * idx_ + 1) - 1;
            int idx_r_ = offset_ * (2 * idx_ + 2) - 1;
            sh_data_[idx_r_] += sh_data_[idx_l_];
        }

        offset_ <<= 1;
        __syncthreads();
    }

    if (idx_ == 0)
        sh_data_[n-1] = 0;
    __syncthreads();

    for (unsigned int d = 1; d < n; d <<= 1)
    {
        offset_ >>= 1;

        if (idx_ < d)
        {
            int idx_l_ = offset_ * (2 * idx_ + 1) - 1;
            int idx_r_ = offset_ * (2 * idx_ + 2) - 1;

            float t_ = sh_data_[idx_l_];
            sh_data_[idx_l_] = sh_data_[idx_r_];
            sh_data_[idx_r_] += t_;
        }
        __syncthreads();
    }

    output[idx_] = sh_data_[idx_];
}

#endif
