/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A GPU memory allocator that implements a 'best-fit with coalescing'
// algorithm.
class GPUBFCAllocator : public BFCAllocator {
 public:
  // See BFCAllocator::Options.
  struct Options {
    // Overridden by TF_FORCE_GPU_ALLOW_GROWTH if that envvar is set.
    bool allow_growth = false;

    // If nullopt, defaults to TF_ENABLE_GPU_GARBAGE_COLLECTION, or true if that
    // envvar is not present.
    //
    // Note:
    //
    //  - BFCAllocator defaults garbage_collection to false, not true.
    //  - this is not the same override behavior as TF_FORCE_GPU_ALLOW_GROWTH.
    absl::optional<bool> garbage_collection;

    double fragmentation_fraction = 0;
    bool allow_retry_on_failure = true;

    // Set shared pool.
    bool share_memory_pool = false;
    mutex* shared_pool_lock = nullptr;
    int64_t* shared_pool_bytes = nullptr;
  };

  GPUBFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
                  const string& name, double fragmentation_fraction = 0.0);
  GPUBFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
                  const GPUOptions& gpu_options, const string& name,
                  double fragmentation_fraction = 0.0);
  GPUBFCAllocator(std::unique_ptr<SubAllocator> sub_allocator, size_t total_memory,
                  const string& name, const Options& opts);
  ~GPUBFCAllocator() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(GPUBFCAllocator);

 private:
  static bool GetAllowGrowthValue(const GPUOptions& gpu_options);
  static bool GetGarbageCollectionValue();
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
