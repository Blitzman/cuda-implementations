#include <functional>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "parallel_reduction.cuh"

template <typename T>
T cpu_reduction(const std::vector<T> & input, std::function<T(T,T)> op, T acc)
{
  for (int i = 0; i < input.size(); ++i)
    acc = op(acc, input[i]);

  return acc;
}

int main(void)
{
  std::random_device random_device_;
  std::mt19937 generator_(random_device_());
  std::uniform_real_distribution<float> distribution_(-1.0, 1.0);

  const int kNumElements = 25600;
  const int kNumBytes = kNumElements * sizeof(float);

  std::cout << "Generating random vector in range [-1.0f, 1.0f] of " << kNumElements << " elements...\n";

  std::vector<float> h_input_(kNumElements);
  for (int i = 0; i < h_input_.size(); ++i)
    h_input_[i] = distribution_(generator_);

  // --- CPU ---------------------------------------------------------------- //

  std::cout << "Executing sum reduction in CPU...\n";

  std::function<float(float,float)> sum_operator_ = [](float a, float b) { return a+b; };
  float result_ = cpu_reduction<float>(h_input_, sum_operator_, 0.0f);

  std::cout << "Result is: " << result_ << "\n";

  // --- GPU ---------------------------------------------------------------- //

  std::cout << "Executing sum reduction in GPU...\n";
  
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

  reduce_kernel<<<bpg_, tpb_>>>(d_input_, d_intermediate_);
  reduce_kernel<<<1, tpb_>>>(d_intermediate_, d_output_);

  cudaMemcpy(&h_output_, d_output_, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input_);
  cudaFree(d_intermediate_);
  cudaFree(d_output_);

  cudaDeviceReset();

  std::cout << "Result is: " << h_output_ << "\n";
}
