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

#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"

#include <cstring>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_stream_aware_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static bool UseCudaMallocAllocator() {
  const char* allocator_env = std::getenv("TF_GPU_ALLOCATOR");
  return allocator_env != nullptr &&
         std::strcmp(allocator_env, "cuda_malloc") == 0;
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static bool UseCudaMemoryGuardAllocator() {
  const char* allocator_env = std::getenv("TF_GPU_ALLOCATOR");
  return allocator_env != nullptr &&
         std::strcmp(allocator_env, "memory_guard") == 0;
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static bool UseCudaMallocAsyncAllocator() {
  const char* allocator_env = std::getenv("TF_GPU_ALLOCATOR");
  auto result = allocator_env != nullptr &&
                std::strcmp(allocator_env, "cuda_malloc_async") == 0;
#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  return result;
#else
  if (result)
    LOG(ERROR) << "TF_GPU_ALLOCATOR=cuda_malloc_async environment found, "
               << "but TensorFlow was not compiled with CUDA 11.2+.";
  return false;
#endif
}

static const bool share_memory_pool = [] {
  bool share_memory_pool;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_GPU_STREAM_GROUP_SHARE_MEM_POOL",
                                      /*default_val=*/false,
                                      &share_memory_pool));
  return share_memory_pool;
}();

/*static*/ GPUProcessState* GPUProcessState::singleton(GPUProcessState* ps) {
  static GPUProcessState* instance = ps ? ps : new GPUProcessState;
  DCHECK((!ps) || (ps == instance))
      << "Multiple calls to GPUProcessState with non-null ps";
  return instance;
}

GPUProcessState::GPUProcessState() : gpu_device_enabled_(false) {
  process_state_ = ProcessState::singleton();
}

int GPUProcessState::BusIdForGPU(TfDeviceId tf_device_id) {
  // Return the NUMA node associated with the GPU's StreamExecutor.
  se::StreamExecutor* se = DeviceIdUtil::ExecutorForTfDeviceId(
                               DEVICE_GPU, GPUMachineManager(), tf_device_id)
                               .ValueOrDie();
  int numa_node = se->GetDeviceDescription().numa_node();
  // bus_id must be non-negative.  If the numa_node is not known,
  // use 0.
  return numa_node >= 0 ? numa_node : 0;
}

// NOLINTNEXTLINE: clang-tidy complains this is unused because of build flags.
static std::unique_ptr<SubAllocator> CreateSubAllocator(
    const GPUOptions& options, PlatformDeviceId platform_device_id,
    const std::vector<SubAllocator::Visitor>& alloc_visitors,
    size_t total_bytes, const std::vector<TfDeviceId>& peer_gpu_ids,
    int stream_id) {
  auto executor = DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                            platform_device_id, stream_id)
                      .ValueOrDie();

  // FIXME(imintz): Observed OOM issues when using the virtual memory
  // allocators. This should be reenabled when resolved.
#if 0 && defined(GOOGLE_CUDA) && CUDA_VERSION >= 10020
  // Use the old allocator when unified memory is required.
  // TODO(imintz): Remove the cuMemAlloc capability of this allocator.
  if (options.per_process_gpu_memory_fraction() > 1.0 ||
      options.experimental().use_unified_memory()) {
    return new DeviceMemAllocator(executor, platform_device_id,
                                  /*use_unified_memory=*/true, alloc_visitors,
                                  {});
  } else {
    auto* gpu_context = reinterpret_cast<stream_executor::gpu::GpuContext*>(
        executor->implementation()->GpuContextHack());

    absl::flat_hash_set<PlatformDeviceId> platform_peer_gpu_ids;
    platform_peer_gpu_ids.reserve(peer_gpu_ids.size());
    for (const TfDeviceId tf_device_id : peer_gpu_ids) {
      PlatformDeviceId platform_device_id;
      TF_CHECK_OK(GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
      platform_peer_gpu_ids.insert(platform_device_id);
    }
    std::vector<PlatformDeviceId> platform_peer_gpu_ids_vec(
        platform_peer_gpu_ids.begin(), platform_peer_gpu_ids.end());

    // Adjust virtual address space to be slightly larger than the physical
    // address space in case the BFC allocator performs suboptimal garbage
    // collection.
    // TODO(imintz): Update BFC allocator to ensure it doesn't create holes in
    // the va space.
    return GpuVirtualMemAllocator::Create(
               alloc_visitors, {}, *gpu_context, platform_device_id,
               /*virtual_address_space_size=*/total_bytes * 2,
               platform_peer_gpu_ids_vec)
        .ValueOrDie()
        .release();
  }
#else
  return absl::WrapUnique(new DeviceMemAllocator(
      executor, platform_device_id,
      (options.per_process_gpu_memory_fraction() > 1.0 ||
       options.experimental().use_unified_memory()),
      alloc_visitors, {}));
#endif
}

Allocator* GPUProcessState::GetGPUAllocator(
    const GPUOptions& options, TfDeviceId tf_device_id, size_t total_bytes,
    const std::vector<TfDeviceId>& peer_gpu_ids, int stream_id) {
  CHECK(process_state_);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  const string& allocator_type = options.allocator_type();
  mutex_lock lock(mu_);
  DeviceIdUtil::CheckValidTfDeviceId(DEVICE_GPU, GPUMachineManager(),
                                     tf_device_id);

  if (tf_device_id.value() >= static_cast<int64_t>(gpu_allocators_.size())) {
    gpu_allocators_.resize(tf_device_id.value() + 1);
  }
  if (stream_id >=
      static_cast<int>(gpu_allocators_[tf_device_id.value()].size())) {
    gpu_allocators_[tf_device_id.value()].resize(stream_id + 1);
  }

  AllocatorParts& allocator_parts =
      gpu_allocators_[tf_device_id.value()][stream_id];
  if (allocator_parts.allocator == nullptr) {
    // Validate allocator types.
    if (!allocator_type.empty() && allocator_type != "BFC") {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }

    PlatformDeviceId platform_device_id;
    TF_CHECK_OK(
        GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
    int bus_id = BusIdForGPU(tf_device_id);
    DCHECK_GE(bus_id, 0);
    while (bus_id >= gpu_visitors_.size()) {
      gpu_visitors_.push_back({});
    }
    while (stream_id >= gpu_visitors_[bus_id].size()) {
      gpu_visitors_[bus_id].push_back({});
    }
    std::unique_ptr<SubAllocator> sub_allocator = CreateSubAllocator(
        options, platform_device_id, gpu_visitors_[bus_id][stream_id],
        total_bytes, peer_gpu_ids, stream_id);
    SubAllocator* sub_allocator_ptr = sub_allocator.get();

    if (share_memory_pool && shared_pool_bytes_.find(tf_device_id.value()) ==
                                 shared_pool_bytes_.end()) {
      shared_pool_lock_.emplace(tf_device_id.value(), mutex());
      shared_pool_bytes_.emplace(tf_device_id.value(), 0);
    }

    auto gpu_bfc_allocator = absl::make_unique<GPUBFCAllocator>(
        std::move(sub_allocator), total_bytes,
        strings::StrCat("GPU_", tf_device_id.value(), "_", stream_id, "_bfc"),
        [&] {
          GPUBFCAllocator::Options o;
          o.allow_growth = options.allow_growth();
          o.allow_retry_on_failure =
              !options.experimental().disallow_retry_on_allocation_failure();
          o.fragmentation_fraction =
              options.experimental().internal_fragmentation_fraction();
          o.share_memory_pool = share_memory_pool;
          o.shared_pool_lock = &shared_pool_lock_[tf_device_id.value()];
          o.shared_pool_bytes = &shared_pool_bytes_[tf_device_id.value()];
          return o;
        }());
    Allocator* gpu_allocator = gpu_bfc_allocator.get();

    SharedCounter* timing_counter = nullptr;
    if (options.experimental().timestamped_allocator()) {
      timing_counter = new SharedCounter;
      gpu_bfc_allocator->SetTimingCounter(timing_counter);
    }

    // If true, checks for memory overwrites by writing
    // distinctive patterns on both ends of allocated memory.
    if (UseCudaMemoryGuardAllocator()) {
      LOG(INFO) << "Using memory guard allocator for GPU.";
      gpu_allocator = new GPUDebugAllocator(gpu_allocator, platform_device_id, stream_id);
      gpu_allocator =
          new GPUNanResetAllocator(gpu_allocator, platform_device_id, stream_id);
    } else if (UseCudaMallocAllocator()) {
      LOG(INFO) << "Using CUDA malloc allocator for GPU.";
      // If true, passes all allocation requests through to cudaMalloc
      // useful for doing memory debugging with tools like cuda-memcheck
      // **WARNING** probably will not work in a multi-gpu scenario
      gpu_bfc_allocator.reset();
      gpu_allocator = new GPUcudaMallocAllocator(platform_device_id, stream_id);
    } else if (UseCudaMallocAsyncAllocator() ||
               options.experimental().use_cuda_malloc_async()) {
      LOG(INFO) << "Using CUDA malloc Async allocator for GPU: "
                << platform_device_id << ", stream: " << stream_id;
      // If true, passes all allocation requests through to cudaMallocAsync
      // TODO: useful for doing memory debugging with tools like
      // compute-sanitizer.
      // TODO: **WARNING** probably will not work in a multi-gpu scenario
      gpu_bfc_allocator.reset();
      gpu_allocator =
          new GpuCudaMallocAsyncAllocator(platform_device_id, total_bytes,
                                          false, true, stream_id);
    }

    Allocator* recording_allocator = nullptr;
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::GPU;
      md.dev_index = platform_device_id.value();
      md.gpu_registered = false;
      md.nic_registered = true;
      recording_allocator = new internal::RecordingAllocator(
          &process_state_->mem_desc_map_, gpu_allocator, md, &mu_);
    }
    allocator_parts = {std::unique_ptr<Allocator>(gpu_allocator),
                       std::unique_ptr<SharedCounter>(timing_counter),
                       gpu_bfc_allocator.release(), sub_allocator_ptr,
                       std::unique_ptr<Allocator>(recording_allocator)};
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
    return allocator_parts.recording_allocator.get();
  } else {
    return allocator_parts.allocator.get();
  }
#else
  LOG(FATAL) << "GPUAllocator unavailable. Not compiled with --config=cuda or "
                "--config=rocm.";
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::GetGPUAllocators(
    const GPUOptions& options, TfDeviceId tf_device_id, size_t total_bytes,
    const std::vector<TfDeviceId>& peer_gpu_ids, size_t num_allocators,
    std::vector<Allocator*>& allocators) {
  // Divide the memory by stream group count if async allocator is not used
  // and don't share_memory_pool between stream groups.
  if ((UseCudaMemoryGuardAllocator() || UseCudaMallocAllocator() ||
       (!UseCudaMallocAsyncAllocator() &&
        !options.experimental().use_cuda_malloc_async())) &&
      !share_memory_pool) {
    total_bytes /= num_allocators;
  }
  allocators.resize(num_allocators);
  for (int i = 0; i < num_allocators; ++i) {
    allocators[i] =
        GetGPUAllocator(options, tf_device_id, total_bytes, peer_gpu_ids, i);
  }
}

SharedCounter* GPUProcessState::GPUAllocatorCounter(TfDeviceId tf_device_id,
                                                    int stream_id) {
  DCHECK(process_state_);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  DeviceIdUtil::CheckValidTfDeviceId(DEVICE_GPU, GPUMachineManager(),
                                     tf_device_id);
  mutex_lock l(mu_);
  if (tf_device_id.value() >= static_cast<int64_t>(gpu_allocators_.size())) {
    LOG(ERROR) << "Asked for counter for GPU allocator " << tf_device_id.value()
               << " but only have " << gpu_allocators_.size();
    return nullptr;
  }

  if (stream_id >=
      static_cast<int>(gpu_allocators_[tf_device_id.value()].size())) {
    LOG(ERROR) << "Asked for counter for GPU allocator " << tf_device_id.value()
               << " stream group" << stream_id << " but only have "
               << gpu_allocators_[tf_device_id.value()].size();
    return nullptr;
  }

  AllocatorParts& allocator_parts =
      gpu_allocators_[tf_device_id.value()][stream_id];
  if (allocator_parts.counter.get() == nullptr) {
    if (allocator_parts.bfc_allocator == nullptr) {
      return nullptr;
    }
    SharedCounter* timing_counter = new SharedCounter;
    allocator_parts.bfc_allocator->SetTimingCounter(timing_counter);
    allocator_parts.counter.reset(timing_counter);
  }
  return allocator_parts.counter.get();
#else
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

Allocator* GPUProcessState::GetGpuHostAllocator(int numa_node,
                                                int stream_id) {
  CHECK(process_state_);
  if (!HasGPUDevice() ||
      !process_state_->ProcessState::FLAGS_brain_mem_reg_gpu_dma) {
    return process_state_->GetCPUAllocator(numa_node);
  }
  if (numa_node == port::kNUMANoAffinity) {
    numa_node = 0;
  }
  {
    // Here we optimize the most common use case where gpu_host_allocators_
    // have already been populated and since we're only reading
    // these vectors, we can get by with a shared lock. In the slower case,
    // we take a unique lock and populate these vectors.
    tf_shared_lock lock(mu_);

    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types &&
        !gpu_host_allocators_.empty() &&
        static_cast<int>(gpu_host_allocators_[0].size()) > stream_id &&
        gpu_host_allocators_[0][stream_id].recording_allocator != nullptr) {
      return gpu_host_allocators_[0][stream_id].recording_allocator.get();
    }
    if (static_cast<int>(gpu_host_allocators_.size()) > numa_node &&
        static_cast<int>(gpu_host_allocators_[0].size()) > stream_id &&
        gpu_host_allocators_[0][stream_id].allocator != nullptr) {
      return gpu_host_allocators_[0][stream_id].allocator.get();
    }
  }

  mutex_lock lock(mu_);
  // Find the first valid StreamExecutor to request CUDA or ROCm host memory
  // through, since any will work.
  //
  // This search isn't super clean, and it would be nice to use a
  // better source of information about which executor to use.  For
  // example, process_state could maybe save the first stream executor
  // it knows is valid.
  se::StreamExecutor* se = nullptr;
  for (int i = 0; i < static_cast<int>(gpu_allocators_.size()); ++i) {
    for (int j = 0; j < static_cast<int>(gpu_allocators_[i].size()); ++j) {
      if (gpu_allocators_[i][j].allocator != nullptr) {
        se = DeviceIdUtil::ExecutorForTfDeviceId(DEVICE_GPU, GPUMachineManager(),
                                                TfDeviceId(i), j)
                .ValueOrDie();
        break;
      }
    }
    if (se != nullptr) break;
  }

  CHECK_NE(nullptr, se);

  while (static_cast<int>(gpu_host_allocators_.size()) <= numa_node) {
    gpu_host_allocators_.push_back({});
  }
  while (gpu_host_alloc_visitors_.size() <= numa_node) {
    gpu_host_alloc_visitors_.push_back({});
  }
  for (int n = 0; n <= numa_node; ++n) {
    while (gpu_host_alloc_visitors_[n].size() <= stream_id) {
      gpu_host_alloc_visitors_[n].push_back({});
    }
  }
  while (gpu_host_free_visitors_.size() <= numa_node) {
    gpu_host_free_visitors_.push_back({});
  }
  for (int n = 0; n <= numa_node; ++n) {
    while (gpu_host_free_visitors_[n].size() <= stream_id) {
      gpu_host_free_visitors_[n].push_back({});
    }
  }
  // Create one allocator for every numa node at the given stream.
  for (int numa_idx = 0; numa_idx <= numa_node; ++numa_idx) {
    if (static_cast<int>(gpu_host_allocators_[numa_idx].size()) > stream_id &&
        gpu_host_allocators_[numa_idx][stream_id].allocator != nullptr) {
      continue;
    }

    SubAllocator* sub_allocator = new DeviceHostAllocator(
        se, numa_idx, gpu_host_alloc_visitors_[numa_idx][stream_id],
        gpu_host_free_visitors_[numa_idx][stream_id]);
    // TODO(zheng-xq): evaluate whether 64GB by default is the best choice.
    int64_t gpu_host_mem_limit_in_mb = -1;
    Status status = ReadInt64FromEnvVar("TF_GPU_HOST_MEM_LIMIT_IN_MB",
                                        1LL << 16 /*64GB max by default*/,
                                        &gpu_host_mem_limit_in_mb);
    if (!status.ok()) {
      LOG(ERROR) << "GetGpuHostAllocator: " << status.error_message();
    }
    int64_t gpu_host_mem_limit = gpu_host_mem_limit_in_mb * (1LL << 20);

    BFCAllocator::Options allocator_opts;
    allocator_opts.allow_growth = true;
    Allocator* allocator = 
        new BFCAllocator(
          absl::WrapUnique(sub_allocator), static_cast<size_t>(gpu_host_mem_limit),
          /*allow_growth=*/true,
          /*name=*/strings::StrCat("gpu_host_", stream_id, "_bfc"),
          allocator_opts);

    if (LogMemory::IsEnabled() && !allocator->TracksAllocationSizes()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingAllocator(allocator, true);
    }
    while (static_cast<int>(gpu_host_allocators_[numa_idx].size()) <=
           stream_id) {
      gpu_host_allocators_[numa_idx].push_back({});
    }
    gpu_host_allocators_[numa_idx][stream_id] = {
        std::unique_ptr<Allocator>(allocator),
        std::unique_ptr<SharedCounter>(nullptr), nullptr, sub_allocator,
        std::unique_ptr<Allocator>(nullptr)};
    AllocatorParts& allocator_parts = gpu_host_allocators_[numa_idx][stream_id];
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::CPU;
      md.dev_index = 0;
      md.gpu_registered = true;
      md.nic_registered = false;
      allocator_parts.recording_allocator.reset(
          new internal::RecordingAllocator(&process_state_->mem_desc_map_,
                                           allocator_parts.allocator.get(), md,
                                           &mu_));
    }
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
    return gpu_host_allocators_[0][stream_id].recording_allocator.get();
  } else {
    return gpu_host_allocators_[0][stream_id].allocator.get();
  }
}

void GPUProcessState::GetGPUStreamAwareAllocators(
    const GPUOptions& options, TfDeviceId tf_device_id, size_t total_bytes,
    const std::vector<TfDeviceId>& peer_gpu_ids, size_t num_allocators,
    std::vector<Allocator*>& allocators) {
  Allocator* stream_aware_bfc = GetGPUStreamAwareAllocator(
      options, tf_device_id, total_bytes, peer_gpu_ids);
  if (tf_device_id.value() >=
      static_cast<int64_t>(gpu_stream_aware_wrappers_.size())) {
    gpu_stream_aware_wrappers_.resize(tf_device_id.value() + 1);
  }
  gpu_stream_aware_wrappers_[tf_device_id.value()].resize(num_allocators);
  allocators.resize(num_allocators);
  for (int i = 0; i < num_allocators; ++i) {
    auto wrapper = absl::make_unique<GPUStreamAwareBFCWrapperAllocator>(
        strings::StrCat("GPU_", tf_device_id.value(),
                        "_stream_aware_bfc_wrapper"),
        static_cast<GPUStreamAwareBFCAllocator*>(stream_aware_bfc));
    allocators[i] = wrapper.get();
    // don't move the above line below. Cause after std::move, it's empty
    gpu_stream_aware_wrappers_[tf_device_id.value()][i] = std::move(wrapper);
  }
}
Allocator* GPUProcessState::GetGPUStreamAwareAllocator(
    const GPUOptions& options, TfDeviceId tf_device_id, size_t total_bytes,
    const std::vector<TfDeviceId>& peer_gpu_ids) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  const string& allocator_type = options.allocator_type();
  mutex_lock lock(mu_);
  DeviceIdUtil::CheckValidTfDeviceId(
      DEVICE_GPU, GPUMachineManager(), tf_device_id);
  if (tf_device_id.value() >=
      static_cast<int64_t>(gpu_stream_aware_allocators_.size())) {
    gpu_stream_aware_allocators_.resize(tf_device_id.value() + 1);
  }

  Allocator* allocator =
      gpu_stream_aware_allocators_[tf_device_id.value()].get();
  if (allocator == nullptr) {
    // Validate allocator types.
    if (!allocator_type.empty() && allocator_type != "BFC") {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }
    PlatformDeviceId platform_device_id;
    TF_CHECK_OK(
        GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
    int bus_id = BusIdForGPU(tf_device_id);
    DCHECK_GE(bus_id, 0);
    while (bus_id >= gpu_stream_aware_visitors_.size()) {
      gpu_stream_aware_visitors_.push_back({});
    }
    std::unique_ptr<SubAllocator> sub_allocator = CreateSubAllocator(
        options, platform_device_id, gpu_stream_aware_visitors_[bus_id],
        total_bytes, peer_gpu_ids, 0);
    auto gpu_bfc_stream_aware_allocator =
        absl::make_unique<GPUStreamAwareBFCAllocator>(
            std::move(sub_allocator), total_bytes,
            strings::StrCat("GPU_", tf_device_id.value(), "_stream_aware_bfc"),
            [&] {
              BFCAllocator::Options o;
              o.allow_growth = options.allow_growth();
              o.allow_retry_on_failure =
                  !options.experimental()
                       .disallow_retry_on_allocation_failure();
              o.fragmentation_fraction =
                  options.experimental().internal_fragmentation_fraction();
              return o;
            }());
    allocator = gpu_bfc_stream_aware_allocator.get();
    gpu_stream_aware_allocators_[tf_device_id.value()] =
        std::move(gpu_bfc_stream_aware_allocator);
    // SharedCounter* timing_counter = nullptr;
    // GPUStreamAwareBFC disable timestamped functionality
    // if (options.experimental().timestamped_allocator()) {
    //   timing_counter = new SharedCounter;
    //   gpu_bfc_stream_aware_allocator->SetTimingCounter(timing_counter);
    // }
    Allocator* recording_allocator = nullptr;
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::GPU;
      md.dev_index = platform_device_id.value();
      md.gpu_registered = false;
      md.nic_registered = true;
      recording_allocator = new internal::RecordingAllocator(
          &process_state_->mem_desc_map_, allocator, md, &mu_);
      return recording_allocator;
    }
  }
  return allocator;
#else
  LOG(FATAL) << "GPUStreamAwareAllocator unavailable. Not compiled with "
                "--config=cuda or "
                "--config=rocm.";
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::AddGPUAllocVisitor(int bus_id,
                                         const SubAllocator::Visitor& visitor,
                                         int stream_id) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  mutex_lock lock(mu_);
  CHECK(gpu_allocators_.empty())  // Crash OK
      << "AddGPUAllocVisitor must be called before "
         "first call to GetGPUAllocator.";
  DCHECK_GE(bus_id, 0);
  while (bus_id >= static_cast<int64_t>(gpu_visitors_.size())) {
    gpu_visitors_.push_back(std::vector<std::vector<SubAllocator::Visitor>>());
  }
  while (stream_id >= static_cast<int>(gpu_visitors_[bus_id].size())) {
    gpu_visitors_[bus_id].push_back(std::vector<SubAllocator::Visitor>());
  }
  gpu_visitors_[bus_id][stream_id].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::AddGpuHostAllocVisitor(
    int numa_node, const SubAllocator::Visitor& visitor, int stream_id) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  mutex_lock lock(mu_);
  CHECK(gpu_host_allocators_.empty())  // Crash OK
      << "AddGpuHostAllocVisitor must be called before "
         "first call to GetGpuHostAllocator.";
  while (numa_node >= static_cast<int64_t>(gpu_host_alloc_visitors_.size())) {
    gpu_host_alloc_visitors_.push_back(
        std::vector<std::vector<SubAllocator::Visitor>>());
  }
  while (stream_id >=
         static_cast<int>(gpu_host_alloc_visitors_[numa_node].size())) {
    gpu_host_alloc_visitors_[numa_node].push_back(
        std::vector<SubAllocator::Visitor>());
  }
  gpu_host_alloc_visitors_[numa_node][stream_id].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::AddGpuHostFreeVisitor(
    int numa_node, const SubAllocator::Visitor& visitor, int stream_id) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  mutex_lock lock(mu_);
  CHECK(gpu_host_allocators_.empty())  // Crash OK
      << "AddGpuHostFreeVisitor must be called before "
         "first call to GetGpuHostAllocator.";
  while (numa_node >= static_cast<int64_t>(gpu_host_free_visitors_.size())) {
    gpu_host_free_visitors_.push_back(
        std::vector<std::vector<SubAllocator::Visitor>>());
  }
  while (stream_id >=
         static_cast<int>(gpu_host_free_visitors_[numa_node].size())) {
    gpu_host_free_visitors_[numa_node].push_back(
        std::vector<SubAllocator::Visitor>());
  }
  gpu_host_free_visitors_[numa_node][stream_id].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::TestOnlyReset() {
  if (process_state_) {
    process_state_->ProcessState::TestOnlyReset();
  }
  {
    mutex_lock lock(mu_);
    gpu_device_enabled_ = false;
    gpu_allocators_.clear();
    gpu_stream_aware_allocators_.clear();
    gpu_visitors_.clear();
    gpu_host_allocators_.clear();
    gpu_host_alloc_visitors_.clear();
    gpu_host_free_visitors_.clear();
  }
}

}  // namespace tensorflow