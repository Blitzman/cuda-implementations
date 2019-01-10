#include <functional>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "parallel_scan.cuh"

void cpu_inclusive_scan(const std::vector<int> & input, std::vector<int> & rOutput)
{
  int acc_ = input[0];
  rOutput[0] = acc_;

  for (int i = 1; i < input.size(); ++i)
  {
    acc_ += input[i];
    rOutput[i] = acc_;
  }
}

int main(void)
{
  std::random_device random_device_;
  std::mt19937 generator_(random_device_());
  std::uniform_int_distribution<unsigned int> distribution_(0, 255);

  const int kNumElements = 1024;
  const int kNumBytes = kNumElements * sizeof(int);

  std::cout << "Generating random vector in range [0, 255] of " << kNumElements << " elements...\n";

  std::vector<int> h_input_(kNumElements);
  for (int i = 0; i < h_input_.size(); ++i)
    h_input_[i] = distribution_(generator_);

  std::cout << "Vector is: \n";
  for (int i = 0; i < h_input_.size(); ++i)
    std::cout << "[" << i << "]:" << h_input_[i] << " ";
  std::cout << "\n";

  // --- CPU ---------------------------------------------------------------- //

  std::cout << "Executing prefix scan in CPU...\n";

  std::vector<int> h_output_(kNumElements);
  cpu_inclusive_scan(h_input_, h_output_);

  std::cout << "Result is: \n";
  for (int i = 0; i < h_output_.size(); ++i)
    std::cout << "[" << i << "]:" << h_output_[i] << " ";
  std::cout << "\n";

  // --- GPU ---------------------------------------------------------------- //

  std::cout << "Executing prefix scan in GPU...\n";
  
  const int threads_per_block_ = kNumElements;
  const int blocks_per_grid_ = 1;
  
  cudaSetDevice(0);

  int *d_input_;
  int *d_output_;
  std::vector<int> h_doutput_(kNumElements);

  cudaMalloc((void**)&d_input_, kNumBytes);
  cudaMalloc((void**)&d_output_, kNumBytes);

  cudaMemcpy(d_input_, h_input_.data(), kNumBytes, cudaMemcpyHostToDevice);

  dim3 tpb_(threads_per_block_, 1, 1);
  dim3 bpg_(blocks_per_grid_, 1, 1);

  std::cout << "Threads Per Block: " << tpb_.x << "\n";
  std::cout << "Blocks Per Grid: " << bpg_.x << "\n";

  // Naive GPU implementation
  //gpu_scan_hillissteele<<<bpg_, tpb_, kNumBytes>>>(d_input_, d_output_, kNumElements);
  gpu_scan_blelloch<<<bpg_, tpb_, kNumBytes>>>(d_input_, d_output_, kNumElements);

  cudaMemcpy(h_doutput_.data(), d_output_, kNumBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input_);
  cudaFree(d_output_);

  cudaDeviceReset();

  std::cout << "Result is: \n";
  for (int i = 0; i < h_doutput_.size(); ++i)
    std::cout << "[" << i << "]:" << h_doutput_[i] << " ";
  std::cout << "\n";

  // Check results

  for (int i = 0; i < h_output_.size(); ++i)
  {
    if (h_output_[i] != h_doutput_[i])
    {
      std::cout << "Found discrepancy at " << i << "\n";
      break;
    }
  }
}
