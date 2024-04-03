#pragma once

#include <tiny-cuda-nn/gpu_matrix.h>

#include <memopt.hpp>

#include <functional>

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

typedef std::function<void(std::map<void *, void *>, cudaStream_t)> Task;

inline std::vector<Task> tasks;

inline void register_and_execute_task(
  std::vector<void *> inputs,
  std::vector<void *> outputs,
  Task task,
  cudaStream_t stream
) {
  auto taskId = tasks.size();
  memopt::annotateNextTask(taskId, inputs, outputs, stream);
  task({}, stream);
}

}  // namespace memopt_adapter
