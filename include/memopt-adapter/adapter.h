#pragma once

#include <tiny-cuda-nn/gpu_matrix.h>

#include <functional>
#include <memopt.hpp>

namespace memopt_adapter {

inline std::vector<tcnn::GPUMatrixBase *> managedMatrices;

template <typename T>
void register_array(tcnn::GPUMatrixDynamic<T> &matrix, bool input = false, bool output = false) {
  managedMatrices.push_back(&matrix);

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

typedef std::function<void(cudaStream_t)> Task;

inline std::vector<Task> tasks;

inline void register_and_execute_task(
  std::vector<void *> inputs,
  std::vector<void *> outputs,
  Task task,
  cudaStream_t stream
) {
  auto taskId = tasks.size();
  tasks.push_back(task);
  memopt::annotateNextTask(taskId, inputs, outputs, stream);
  task(stream);
}

inline void execute_random_task(int taskId, std::map<void *, void *> addressUpdateMap, cudaStream_t stream) {
  for (auto matrix : managedMatrices) {
    matrix->try_updating_address(addressUpdateMap);
  }
  tasks[taskId](stream);
}

}  // namespace memopt_adapter
