#include <functional>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "parallel_histogram.cuh"

void cpu_histogram(const std::vector<unsigned int> & input, std::vector<unsigned int> & rHistogram, unsigned int bins)
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
  std::uniform_int_distribution<unsigned int> distribution_(0, 255);

  const unsigned long kNumElements = 32768;
  const unsigned int kNumBytes = kNumElements * sizeof(unsigned int);
  const unsigned int kNumBins = 256;

  std::cout << "Generating random vector in range [0, 255] of " << kNumElements << " elements...\n";

  std::vector<unsigned int> h_input_(kNumElements);
  for (int i = 0; i < h_input_.size(); ++i)
    h_input_[i] = distribution_(generator_);

  // --- CPU ---------------------------------------------------------------- //

  std::cout << "Executing histogram in CPU...\n";

  std::vector<unsigned int> h_histogram_(kNumBins);
  cpu_histogram(h_input_, h_histogram_, kNumBins);

  std::cout << "Result is: \n";
  for (int i = 0; i < kNumBins; ++i)
    std::cout << "[" << i << "]:" << h_histogram_[i] << " ";
  std::cout << "\n";

  // --- GPU ---------------------------------------------------------------- //

  std::cout << "Executing histogram in GPU...\n";
  
  const int threads_per_block_ = 1024;
  const int blocks_per_grid_ = kNumElements / threads_per_block_;
  
  cudaSetDevice(0);

  unsigned int *d_input_;
  unsigned int *d_histogram_;
  
  std::vector<int> h_dhistogram_(kNumBins);

  cudaMalloc((void**)&d_input_, kNumBytes);
  cudaMalloc((void**)&d_histogram_, sizeof(int) * kNumBins); // Overallocated

  cudaMemcpy(d_input_, h_input_.data(), kNumBytes, cudaMemcpyHostToDevice);

  dim3 tpb_(threads_per_block_, 1, 1);
  dim3 bpg_(blocks_per_grid_, 1, 1);

  std::cout << "Threads Per Block: " << tpb_.x << "\n";
  std::cout << "Blocks Per Grid: " << bpg_.x << "\n";

  // Naive GPU implementation
  //gpu_histogram_naive<<<bpg_, tpb_>>>(d_input_, d_histogram_, kNumBins);

  // Atomic GPU implementation
  //gpu_histogram_atomic<<<bpg_, tpb_>>>(d_input_, d_histogram_, kNumBins);

  // Strided GPU implementation
  dim3 tpb_strided_(threads_per_block_, 1, 1);
  dim3 bpg_strided_(256, 1, 1);
  //gpu_histogram_atomic_strided<<<bpg_strided_, tpb_strided_>>>(d_input_, d_histogram_, kNumBins, kNumElements);

  // Strided privatized GPU  implementation
  gpu_histogram_atomic_strided_privatized<<<bpg_strided_, tpb_strided_, kNumBins * sizeof(unsigned int)>>>(d_input_, d_histogram_, kNumBins, kNumElements);

  cudaMemcpy(h_dhistogram_.data(), d_histogram_, sizeof(int) * kNumBins, cudaMemcpyDeviceToHost);

  cudaFree(d_input_);
  cudaFree(d_histogram_);

  cudaDeviceReset();

  std::cout << "Result is: \n";
  for (int i = 0; i < kNumBins; ++i)
    std::cout << "[" << i << "]:" << h_dhistogram_[i] << " ";
  std::cout << "\n";
}
