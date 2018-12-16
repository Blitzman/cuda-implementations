#ifndef KERNEL_PARALLEL_REDUCTION_
#define KERNEL_PARALLEL_REDUCTION_

template <typename T>
__global__ void gpu_reduction_naive(T * input, T * output)
{
  int idx_ = threadIdx.x + blockDim.x * blockIdx.x;

  for (unsigned int s = 1; s < blockDim.x; s <<= 1)
  {
    if (threadIdx.x % (s << 1) == 0)
      input[idx_] += input[idx_ + s];
    
    __syncthreads();
  }
  
  if (threadIdx.x == 0)
    output[blockIdx.x] = input[idx_];
}

template <typename T>
__global__ void gpu_reduction_coalesced(T * input, T * output)
{
  int idx_ = threadIdx.x + blockDim.x * blockIdx.x;

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
      input[idx_] += input[idx_ + s];
    
    __syncthreads();
  }
  
  if (threadIdx.x == 0)
    output[blockIdx.x] = input[idx_];
}

template __global__ void gpu_reduction_coalesced<float>(float*, float*);

#endif
