#include <functional>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "parallel_histogram.cuh"

void cpu_histogram(const std::vector<int> & input, std::vector<int> & rHistogram, int bins)
{
  for (int i = 0; i < input.size(); ++i)
  {
    int bin_ = input[i] % bins;
    rHistogram[bin_]++;
  }
}

int main(void)
{
  std::random_device random_device_;
  std::mt19937 generator_(random_device_());
  std::uniform_int_distribution<int> distribution_(0, 255);

  const int kNumElements = 32768;
  const int kNumBytes = kNumElements * sizeof(int);
  const int kNumBins = 256;

  std::cout << "Generating random vector in range [0, 255] of " << kNumElements << " elements...\n";

  std::vector<int> h_input_(kNumElements);
  for (int i = 0; i < h_input_.size(); ++i)
    h_input_[i] = distribution_(generator_);

  // --- CPU ---------------------------------------------------------------- //

  std::cout << "Executing histogram in CPU...\n";

  std::vector<int> h_histogram_(kNumBins);
  cpu_histogram(h_input_, h_histogram_, kNumBins);

  std::cout << "Result is: \n";
  for (int i = 0; i < kNumBins; ++i)
    std::cout << "[" << i << "]:" << h_histogram_[i] << " ";
  std::cout << "\n";

  // --- GPU ---------------------------------------------------------------- //

  /* std::cout << "Executing sum reduction in GPU...\n";
  
  const int threads_per_block_ = 1024;
  const int blocks_per_grid_ = kNumElements / threads_per_block_;
  
  cudaSetDevice(0);

  float h_output_ = 0.0f;
  float *d_input_;
  float *d_intermediate_;
  float *d_output_;

  cudaMalloc((void**)&d_input_, kNumBytes);
  cudaMalloc((void**)&d_intermediate_, kNumBytes); // Overallocated
  cudaMalloc((void**)&d_output_, sizeof(float));

  cudaMemcpy(d_input_, h_input_.data(), kNumBytes, cudaMemcpyHostToDevice);

  dim3 tpb_(threads_per_block_, 1, 1);
  dim3 bpg_(blocks_per_grid_, 1, 1);

  std::cout << "Threads Per Block: " << tpb_.x << "\n";
  std::cout << "Blocks Per Grid: " << bpg_.x << "\n";

  // Naive GPU implementation
  gpu_reduction_naive<<<bpg_, tpb_>>>(d_input_, d_intermediate_);
  gpu_reduction_naive<<<1, bpg_>>>(d_intermediate_, d_output_);

  // Coalesced GPU implementation
  //gpu_reduction_coalesced<<<bpg_, tpb_>>>(d_input_, d_intermediate_);
  //gpu_reduction_coalesced<<<1, bpg_>>>(d_intermediate_, d_output_);

  // Shared Memory GPU implementation
  //gpu_reduction_shmem<<<bpg_, tpb_, tpb_.x * sizeof(float)>>>(d_input_, d_intermediate_);
  //gpu_reduction_shmem<<<1, bpg_, bpg_.x * sizeof(float)>>>(d_intermediate_, d_output_);

  cudaMemcpy(&h_output_, d_output_, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input_);
  cudaFree(d_intermediate_);
  cudaFree(d_output_);

  cudaDeviceReset();

  std::cout << "Result is: " << h_output_ << "\n"; */
}
