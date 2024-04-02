#pragma once

#include <tiny-cuda-nn/gpu_matrix.h>

#include <memopt.hpp>

namespace memopt_adapter {

template <typename T>
void register_array(tcnn::GPUMatrixDynamic<T> &matrix, bool input = false, bool output = false) {
  memopt::registerManagedMemoryAddress(matrix.data(), matrix.n_bytes());
  if (input) {
    memopt::registerApplicationInput(matrix.data());
  }
  if (output) {
    memopt::registerApplicationOutput(matrix.data());
  }
}

inline size_t total_registered_array_bytes() {
  size_t bytes = 0;
  for (const auto &[_, size] : memopt::MemoryManager::managedMemoryAddressToSizeMap) {
    bytes += size;
  }
  return bytes;
}

}  // namespace memopt_adapter
