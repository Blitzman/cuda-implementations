#include <functional>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime_api.h>

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

  const int kNumElements = 4;
  const int kNumBytes = kNumElements * sizeof(float);

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
}
