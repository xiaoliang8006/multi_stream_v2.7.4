#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_STREAM_AWARE_BFC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_STREAM_AWARE_BFC_ALLOCATOR_H_

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

class GPUStreamAwareBFCAllocator : public BFCAllocator {
 public:
  GPUStreamAwareBFCAllocator(std::unique_ptr<SubAllocator> sub_allocator,
                             size_t total_memory, const string& name,
                             const BFCAllocator::Options& opts);

  void AddComputeStream(const int stream_id, se::Stream* stream) {
    if (compute_streams_.size() <= stream_id)
      compute_streams_.resize(stream_id + 1);
    compute_streams_[stream_id] = stream;
  }

  void* AllocateRaw(size_t unused_alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;

  void* AllocateRawInternalWithRetry(
      size_t unused_alignment, size_t num_bytes,
      const AllocationAttributes& allocation_attr) override;

  void* AllocateRawInternal(size_t unused_alignment, size_t num_bytes,
                            bool dump_log_on_failure, uint64 freed_before,
                            int stream_to_allocate);

  void SplitChunk(BFCAllocator::ChunkHandle h, size_t num_bytes) override;

  void* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes,
                     uint64 freed_before, bool must_not_sync,
                     int stream_to_allocate) TF_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  BFCAllocator::ChunkHandle AllocateChunk()
      TF_EXCLUSIVE_LOCKS_REQUIRED(lock_) override;

  BFCAllocator::ChunkHandle TryToCoalesce(BFCAllocator::ChunkHandle h,
                                          bool ignore_freed_at)
      TF_EXCLUSIVE_LOCKS_REQUIRED(lock_) override;

  void Merge(BFCAllocator::ChunkHandle h1, BFCAllocator::ChunkHandle h2)
      TF_EXCLUSIVE_LOCKS_REQUIRED(lock_) override;

  bool Extend(size_t alignment, size_t rounded_bytes)
      TF_EXCLUSIVE_LOCKS_REQUIRED(lock_) override;

  TF_DISALLOW_COPY_AND_ASSIGN(GPUStreamAwareBFCAllocator);

  std::vector<int> chunk_prev_used_streams_;
  std::vector<se::Stream*> compute_streams_;
  const int non_stream_id_ = -1;  // this chunk has not been used on any stream
};

class GPUStreamAwareBFCWrapperAllocator : public Allocator {
 public:
  GPUStreamAwareBFCWrapperAllocator(
      const string& name, GPUStreamAwareBFCAllocator* stream_aware_bfc)
      : name_(name), stream_aware_bfc_(stream_aware_bfc) {}

  std::string Name() override { return name_; };

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return AllocateRaw(alignment, num_bytes, AllocationAttributes());
  }

  void* AllocateRaw(size_t unused_alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr);

  void SetComputeStream(int stream_id, se::Stream* stream);

  void DeallocateRaw(void* ptr) override;
  bool TracksAllocationSizes() const override;
  size_t RequestedSize(const void* ptr) const override;
  size_t AllocatedSize(const void* ptr) const override;
  int64_t AllocationId(const void* ptr) const override;
  absl::optional<AllocatorStats> GetStats() override;
  bool ClearStats() override;
  void SetSafeFrontier(uint64 count) override;
  AllocatorMemoryType GetMemoryType() const;

 private:
  GPUStreamAwareBFCAllocator* stream_aware_bfc_;
  std::string name_;
  int stream_id_;
  se::Stream* stream_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_STREAM_AWARE_BFC_ALLOCATOR_H_