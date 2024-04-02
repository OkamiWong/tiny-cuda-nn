#pragma once

#include <tiny-cuda-nn/gpu_matrix.h>

#include <cstdio>
#include <map>
#include <set>
#include <vector>

namespace preflight {
inline std::map<tcnn::GPUMatrixBase *, size_t> matrixAddressToSize;
inline std::vector<std::set<tcnn::GPUMatrixBase *>> kernelDataDependencies;
inline bool enabled = true;

inline void registerKernel(cudaStream_t stream, std::vector<tcnn::GPUMatrixBase *> matrices) {
  if (!enabled) {
    return;
  }

  cudaStreamCaptureStatus capture_status;
  CUDA_CHECK_THROW(cudaStreamIsCapturing(stream, &capture_status));
  if (capture_status == cudaStreamCaptureStatusNone) {
    return;
  }

  for (auto matrix : matrices) {
    if (matrix) {
      matrixAddressToSize[matrix] = matrix->n_bytes();
    }
  }

  kernelDataDependencies.emplace_back(matrices.begin(), matrices.end());
}

inline void printResult() {
  if (kernelDataDependencies.size() == 0) {
    return;
  }

  size_t totalArraySize = 0;
  for (const auto &[_, size] : matrixAddressToSize) {
    totalArraySize += size;
  }

  size_t bottleNeckKernelDataDependencySize = 0;
  int count = 0;
  for (const auto &matrices : kernelDataDependencies) {
    size_t size = 0;
    for (auto matrix : matrices) {
      size += matrixAddressToSize[matrix];
    }
    printf("[preflight] kernel[%d]: %.4lf MiB\n", count++, static_cast<double>(size) / 1024.0 / 1024.0);
    bottleNeckKernelDataDependencySize = std::max(bottleNeckKernelDataDependencySize, size);
  }

  count = 0;
  for (const auto &[_, size] : matrixAddressToSize) {
    printf("[preflight] matrix[%d]: %.4lf MiB\n", count++, static_cast<double>(size) / 1024.0 / 1024.0);
  }

  printf(
    "[preflight] Total array size (MiB): %.4lf\n",
    static_cast<double>(totalArraySize) / 1024.0 / 1024.0
  );
  printf(
    "[preflight] Bottleneck kernel data dependency size (MiB): %.4lf\n",
    static_cast<double>(bottleNeckKernelDataDependencySize) / 1024.0 / 1024.0
  );
  printf(
    "[preflight] Bottleneck / Total: %.4lf\n",
    static_cast<double>(bottleNeckKernelDataDependencySize) / static_cast<double>(totalArraySize)
  );
}
}  // namespace preflight
