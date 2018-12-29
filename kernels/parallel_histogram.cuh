#ifndef PARALLEL_HISTOGRAM_CUH_
#define PARALLEL_HISTOGRAM_CUH_

__global__
void gpu_histogram_naive(unsigned int* input, unsigned int* histogram, unsigned int bins)
{
  int idx_ = threadIdx.x + blockIdx.x * blockDim.x;
  int bin_ = input[idx_] % bins;
  histogram[bin_] += 1;
}

__global__
void gpu_histogram_atomic(unsigned int* input, unsigned int* histogram, unsigned int bins)
{
  int idx_ = threadIdx.x + blockIdx.x * blockDim.x;
  int bin_ = input[idx_] % bins;
  atomicAdd(&(histogram[bin_]), 1);
}

__global__
void gpu_histogram_atomic_strided(unsigned int* input, unsigned int* histogram, unsigned int bins, unsigned long n)
{
  int idx_ = threadIdx.x + blockIdx.x * blockDim.x;
  int stride_ = blockDim.x * gridDim.x;

  for (unsigned long i = idx_; i < n; i += stride_)
  {
    unsigned int bin_ = input[idx_] % bins;
    atomicAdd(&(histogram[bin_]), 1);
  }
}

__global__
void gpu_histogram_atomic_strided_privatized(unsigned int* input, unsigned int* histogram, unsigned int bins, unsigned long n)
{
  extern __shared__ unsigned int histogram_private_[];

  if (threadIdx.x < bins)
    histogram_private_[threadIdx.x] = 0;

  __syncthreads();

  int idx_ = threadIdx.x + blockIdx.x * blockDim.x;
  int stride_ = blockDim.x * gridDim.x;

  for (unsigned long i = idx_; i < n; i += stride_)
  {
    int bin_ = input[idx_] % bins;
    atomicAdd(&(histogram_private_[bin_]), 1);
  }

  __syncthreads();

  if (threadIdx.x < bins)
    atomicAdd(&(histogram[threadIdx.x]), histogram_private_[threadIdx.x]);
}

#endif