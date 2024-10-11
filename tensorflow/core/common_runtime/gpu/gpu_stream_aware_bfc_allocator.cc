#include "tensorflow/core/common_runtime/gpu/gpu_stream_aware_bfc_allocator.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#ifdef TENSORFLOW_MEM_DEBUG
#include "tensorflow/core/platform/stacktrace.h"
#endif
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/protobuf/bfc_memory_map.pb.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#define CUDA_CHECK(call)                                       \
  do {                                                         \
    cudaError_t cudaError = (call);                            \
    if (cudaError != cudaSuccess) {                            \
      printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(cudaError));                   \
    }                                                          \
  } while (0)

namespace tensorflow {

static const bool bfc_custom_log = [] {
  bool log;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_BFC_CUSTOM_LOG",
                                      /*default_val=*/false, &log));
  return log;
}();

// clang-format off
// NOLINTBEGIN
/* sync table
    stream a,  stream b,  FindChunkPtr(a = stream_to_allocate, b = chunk prev stream),  Merge(merge to a, delete b)
    -1,        -1,        forbid,                                                       no sync
    -1,        >=0,       forbid,                                                       sync & b wait a
    >=0,       -1,        no sync & b change to a,                                      no sync
    >=0,       >=0,       sync if a != b & a wait b & b change to a,                    sync if a != b
*/
// NOLINTEND
// clang-format on

// #define ENABLE_DEBUG_LOG
#ifdef ENABLE_DEBUG_LOG
#define DEBUG_LOG VLOG(0)
#else
#define DEBUG_LOG \
  if (false) VLOG(0)
#endif

GPUStreamAwareBFCAllocator::GPUStreamAwareBFCAllocator(
    std::unique_ptr<SubAllocator> sub_allocator, size_t total_memory,
    const string& name, const BFCAllocator::Options& opts)
    : BFCAllocator(std::move(sub_allocator), total_memory,
    /* gpu_memory_allow_growth */ true, name, opts) {}

void* GPUStreamAwareBFCAllocator::AllocateRaw(
    size_t unused_alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
  VLOG(3) << "AllocateRaw " << Name() << "  " << num_bytes;
  void* result = [&] {
    if (!opts_.allow_retry_on_failure || !allocation_attr.retry_on_failure) {
      // If we have globally disabled retry-on-failure and fail to allocate an
      // "important" alloc, we want to print a log, because the program may be
      // about to fail due to OOM.
      //
      // Bit of a hack: We deem "important" allocs as those which are retryable.
      // In TF, *non*-retryable allocations are usually those which we can
      // tolerate failing.  For example, we allocate convolution scratch memory
      // as non-retryable; if it fails, we'll just use a fallback algorithm that
      // uses no scratch.
      static std::atomic<int32> log_counter{0};
      constexpr int kMaxFailureLogs = 10;
      bool dump_log_on_failure =
          (/*retry is globally disabled*/ !opts_.allow_retry_on_failure &&
           /*alloc is "important"*/ allocation_attr.retry_on_failure &&
           log_counter.load(std::memory_order_relaxed) < kMaxFailureLogs) ||
          VLOG_IS_ON(2);

      uint64 freed_by_count = 0;
      if (allocation_attr.freed_by_func != nullptr) {
        freed_by_count = (*allocation_attr.freed_by_func)();
      }
      void* res = AllocateRawInternal(unused_alignment, num_bytes,
                                      dump_log_on_failure, freed_by_count,
                                      allocation_attr.stream_to_allocate);
      if (res == nullptr) {
        int32 counter_value = log_counter.load(std::memory_order_relaxed);
        if (counter_value < kMaxFailureLogs) {
          log_counter.store(counter_value + 1, std::memory_order_relaxed);
          LOG(WARNING)
              << "Allocator (" << Name() << ") ran out of memory trying "
              << "to allocate " << strings::HumanReadableNumBytes(num_bytes)
              << " with freed_by_count=" << freed_by_count << "."
              << (!allocation_attr.retry_on_failure
                      ? " The caller indicates that this is not a failure, but"
                        " this may mean that there could be performance gains "
                        "if more memory were available."
                      : "");
        }
      }
      return res;
    } else {
      return AllocateRawInternalWithRetry(unused_alignment, num_bytes,
                                          allocation_attr);
    }
  }();
  VLOG(3) << "AllocateRaw " << Name() << "  " << num_bytes << " " << result;
  return result;
}

void* GPUStreamAwareBFCAllocator::AllocateRawInternalWithRetry(
    size_t unused_alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
  // Fast path: Try once to allocate without getting the retry_helper_ involved
  uint64 freed_by_count = 0;
  if (allocation_attr.freed_by_func != nullptr) {
    freed_by_count = (*allocation_attr.freed_by_func)();
  }
  void* r =
      AllocateRawInternal(unused_alignment, num_bytes, false, freed_by_count,
                          allocation_attr.stream_to_allocate);
  if (r != nullptr) {
    return r;
  } else {
    static const int64_t kMaxMillisToWait = 10000;  // 10 seconds
    r = retry_helper_.AllocateRaw(
        [this, &allocation_attr](size_t a, size_t nb, bool v) {
          uint64 freed_by_count = 0;
          if (allocation_attr.freed_by_func != nullptr) {
            freed_by_count = (*allocation_attr.freed_by_func)();
          }
          return AllocateRawInternal(a, nb, v, freed_by_count,
                                     allocation_attr.stream_to_allocate);
        },
        kMaxMillisToWait, unused_alignment, num_bytes);
    return r;
  }
}

void* GPUStreamAwareBFCAllocator::AllocateRawInternal(size_t unused_alignment,
                                                      size_t num_bytes,
                                                      bool dump_log_on_failure,
                                                      uint64 freed_before,
                                                      int stream_to_allocate) {
  CHECK(stream_to_allocate >= 0);
  if (num_bytes == 0) {
    VLOG(2) << "tried to allocate 0 bytes";
    return nullptr;
  }

  // First, always allocate memory of at least kMinAllocationSize
  // bytes, and always allocate multiples of kMinAllocationSize bytes
  // so all memory addresses are nicely byte aligned.
  size_t rounded_bytes = RoundedBytes(num_bytes);

  // The BFC allocator tries to find the best fit first.
  BinNum bin_num = BinNumForSize(rounded_bytes);

  mutex_lock l(lock_);
  // if (!timestamped_chunks_.empty()) {
  //   // Merge timestamped chunks whose counts have become safe for general
  //   use. MergeTimestampedChunks(0);
  // }
  DEBUG_LOG << "============\nstream_to_allocate: " << stream_to_allocate
            << ", bin_num: " << bin_num << ", rounded_bytes: " << rounded_bytes;
  void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before,
                           true, stream_to_allocate);
  if (ptr != nullptr) {
    DEBUG_LOG << "Found on same stream";
    AddTraceMe("MemoryAllocation", ptr);
    return ptr;
  }

  // Try to extend
  if (Extend(unused_alignment, rounded_bytes)) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before, true,
                       stream_to_allocate);
    if (ptr != nullptr) {
      DEBUG_LOG << "After extend, found on same stream";
      AddTraceMe("MemoryAllocation", ptr);
      return ptr;
    }
  }

  // free to search on other streams
  ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before, false,
                     stream_to_allocate);
  if (ptr != nullptr) {
    DEBUG_LOG << "Found on other stream";
    AddTraceMe("MemoryAllocation", ptr);
    return ptr;
  }

  // if ((freed_before == 0) && (!timestamped_chunks_.empty())) {
  //   // We're unable to satisfy an allocation request without a specific
  //   // timestamp requirement.  Rather than fail, try merging any held-out
  //   // timestamped chunks more aggressively until a free chunk of the
  //   necessary
  //   // size is formed.
  //   if (MergeTimestampedChunks(rounded_bytes)) {
  //     ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before,
  //     stream_to_allocate); if (ptr != nullptr) {
  //       AddTraceMe("MemoryAllocation", ptr);
  //       return ptr;
  //     }
  //   }
  // }

  // Reaching this point means that no chunks can satisfy the request. Also,
  // the unallocated bytes cannot satisfy the request. Before giving up, let's
  // try deallocating free regions so that suballocator can combine them with
  // the unallocated bytes and form a larger region.
  if (DeallocateFreeRegions(rounded_bytes) &&
      Extend(unused_alignment, rounded_bytes)) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before, false,
                       stream_to_allocate);
    if (ptr != nullptr) {
      DEBUG_LOG << "After deallocate and extend, found on all stream";
      AddTraceMe("MemoryAllocation", ptr);
      return ptr;
    }
  }

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  // Dump the memory log for analysis.
  MaybeWriteMemoryMap();
  if (dump_log_on_failure) {
    LOG(WARNING)
        << "Allocator (" << Name() << ") ran out of memory trying "
        << "to allocate " << strings::HumanReadableNumBytes(num_bytes)
        << " (rounded to " << rounded_bytes << ")"
        << "requested by op "
        << tensorflow::profiler::ScopedMemoryDebugAnnotation::
               CurrentAnnotation()
                   .pending_op_name
        << "\nIf the cause is memory fragmentation maybe the environment "
        << "variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will "
        << "improve the situation. \nCurrent allocation summary follows."
        << "\nCurrent allocation summary follows.";
    DumpMemoryLog(rounded_bytes);
    LOG(WARNING) << RenderOccupancy();
  }
  return nullptr;
}

void* GPUStreamAwareBFCAllocator::FindChunkPtr(
    BinNum bin_num, size_t rounded_bytes, size_t num_bytes, uint64 freed_before,
    bool must_not_sync, int stream_to_allocate) {
  // First identify the first bin that could satisfy rounded_bytes.
  CHECK(stream_to_allocate != non_stream_id_);

  for (; bin_num < kNumBins; bin_num++) {
    // Start searching from the first bin for the smallest chunk that fits
    // rounded_bytes.
    DEBUG_LOG << "Looking for bin_num: " << bin_num;
    Bin* b = BinFromIndex(bin_num);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
         ++citer) {
      const BFCAllocator::ChunkHandle h = (*citer);
      BFCAllocator::Chunk* chunk = ChunkFromHandle(h);
      CHECK(!chunk->in_use());

      if (freed_before > 0 && freed_before < chunk->freed_at_count) {
        continue;
      }

      if (chunk->size >= rounded_bytes) {
        int prev_used_stream = chunk_prev_used_streams_[h];

        bool need_sync = stream_to_allocate >= 0 && prev_used_stream >= 0 &&
                         stream_to_allocate != prev_used_stream;
        if (must_not_sync && need_sync) {
          DEBUG_LOG << "Not satisfied must_not_sync: " << must_not_sync
                    << ", need_sync: " << need_sync
                    << ", stream_to_allocate: " << stream_to_allocate
                    << ", prev_used_stream: " << prev_used_stream;
          continue;
        }

        DEBUG_LOG << "Found suitable chunk, stream: " << prev_used_stream
                  << ", size: " << chunk->size
                  << ", requested_size: " << chunk->requested_size
                  << ", in_use: " << chunk->in_use();

        // We found an existing chunk that fits us that wasn't in use, so
        // remove it from the free bin structure prior to using.
        RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

        // If we can break the size of the chunk into two reasonably large
        // pieces, do don't waste more than max_internal_fragmentation_bytes on
        // padding. If this threshold is not set by the user, then use 128MB as
        // the default.
        const int64_t max_internal_fragmentation_bytes =
            (opts_.fragmentation_fraction > 0.0)
                ? opts_.fragmentation_fraction * memory_limit_
                : 128 << 20;

        if (chunk->size >= rounded_bytes * 2 ||
            static_cast<int64_t>(chunk->size) - rounded_bytes >=
                max_internal_fragmentation_bytes) {
          SplitChunk(h, rounded_bytes);
          chunk = ChunkFromHandle(h);  // Update chunk pointer in case it moved
        }

        // The requested size of the returned chunk is what the user
        // has allocated.
        chunk->requested_size = num_bytes;
        // Assign a unique id and increment the id counter, marking the
        // chunk as being in use.
        chunk->allocation_id = next_allocation_id_++;

        // Update stats.
        ++stats_.num_allocs;
        stats_.bytes_in_use += chunk->size;
        if (stats_.bytes_in_use > stats_.peak_bytes_in_use) {
          VLOG(2) << "New Peak memory usage of " << stats_.bytes_in_use
                  << " bytes for " << Name();
        }
        stats_.peak_bytes_in_use =
            std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
        stats_.largest_alloc_size =
            std::max<std::size_t>(stats_.largest_alloc_size, chunk->size);

#ifdef TENSORFLOW_MEM_DEBUG
        if (ShouldRecordOpName()) {
          const auto& annotation =
              profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
          if (annotation.pending_op_name != nullptr) {
            chunk->op_name = annotation.pending_op_name;
          } else {
            LOG(INFO) << "missing pending_op_name for " << Name()
                      << " reading addr "
                      << static_cast<const void*>(&annotation.pending_op_name)
                      << "\n"
                      << CurrentStackTrace();
            chunk->op_name = nullptr;
          }
          chunk->action_count = ++action_counter_;
          chunk->step_id = annotation.pending_step_id;
          int slot = chunk->action_count % MEM_DEBUG_SIZE_HISTORY_SIZE;
          size_history_[slot] = stats_.bytes_in_use;
        }
#endif

        if (need_sync) {
          compute_streams_[stream_to_allocate]->ThenWaitFor(
              compute_streams_[prev_used_stream]);
        }
        chunk_prev_used_streams_[h] = stream_to_allocate;

        VLOG(4) << "Returning: " << chunk->ptr;
        if (VLOG_IS_ON(4)) {
          LOG(INFO) << "A: " << RenderOccupancy();
        }

        DEBUG_LOG << "Return suitable chunk, stream: "
                  << chunk_prev_used_streams_[h] << ", size: " << chunk->size
                  << ", requested_size: " << chunk->requested_size
                  << ", in_use: " << chunk->in_use();

        return chunk->ptr;
      }
    }
  }

  return nullptr;
}

void GPUStreamAwareBFCAllocator::SplitChunk(BFCAllocator::ChunkHandle h,
                                            size_t num_bytes) {
  // Allocate the new chunk before we do any ChunkFromHandle
  ChunkHandle h_new_chunk = AllocateChunk();

  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));

  // Create a new chunk starting num_bytes after c
  BFCAllocator::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

  // Set the new sizes of the chunks.
  new_chunk->size = c->size - num_bytes;
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->allocation_id = -1;

  // It inherits the freed time.
  new_chunk->freed_at_count = c->freed_at_count;

  // It inherits the prev_used_stream;
  chunk_prev_used_streams_[h_new_chunk] = chunk_prev_used_streams_[h];
  DEBUG_LOG << "Split, chunk size: " << c->size
            << ", stream: " << chunk_prev_used_streams_[h]
            << ", new chunk size: " << new_chunk->size
            << ", stream: " << chunk_prev_used_streams_[h_new_chunk];

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  BFCAllocator::ChunkHandle h_neighbor = c->next;
  new_chunk->prev = h;
  new_chunk->next = h_neighbor;
  c->next = h_new_chunk;
  if (h_neighbor != kInvalidChunkHandle) {
    Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
    c_neighbor->prev = h_new_chunk;
  }

  // Add the newly free chunk to the free bin.
  InsertFreeChunkIntoBin(h_new_chunk);
}

BFCAllocator::ChunkHandle GPUStreamAwareBFCAllocator::AllocateChunk() {
  if (free_chunks_list_ != kInvalidChunkHandle) {
    ChunkHandle h = free_chunks_list_;
    Chunk* c = ChunkFromHandle(h);
    free_chunks_list_ = c->next;
    return h;
  } else {
    ChunkHandle h = chunks_.size();
    chunks_.resize(h + 1);
    chunk_prev_used_streams_.resize(h + 1);
    return h;
  }
}

BFCAllocator::ChunkHandle GPUStreamAwareBFCAllocator::TryToCoalesce(
    ChunkHandle h, bool ignore_freed_at) {
  Chunk* c = ChunkFromHandle(h);
  if ((!ignore_freed_at) && c->freed_at_count > 0) return h;
  ChunkHandle coalesced_chunk = h;

  // If the next chunk is free, merge it into c and delete it.
  if (c->next != kInvalidChunkHandle && !ChunkFromHandle(c->next)->in_use()) {
    Chunk* n = ChunkFromHandle(c->next);
    if ((n->freed_at_count == 0) || ignore_freed_at) {
      if (chunk_prev_used_streams_[c->next] != non_stream_id_) {
        // if first chunk is non_stream, don't merge. Maybe cause fragment?
        // Through experiment, found that if can merge cross streams, then
        // easily a large non_stream chunk can be merged by previous chunk, and
        // next time allocation will not find this non_stream chunk, causing
        // extending very greedly, wasting space.
        VLOG(4) << "Merging c->next " << n->ptr << " with c " << c->ptr;
        RemoveFreeChunkFromBin(c->next);
        Merge(h, c->next);
      }
    }
  }

  // If the previous chunk is free, merge c into it and delete c.
  if (c->prev != kInvalidChunkHandle && !ChunkFromHandle(c->prev)->in_use()) {
    Chunk* n = ChunkFromHandle(c->prev);
    if ((n->freed_at_count == 0) || ignore_freed_at) {
      if (chunk_prev_used_streams_[h] != non_stream_id_) {
        // same logic as above
        VLOG(4) << "Merging c " << c->ptr << " into c->prev " << n->ptr;
        coalesced_chunk = c->prev;
        RemoveFreeChunkFromBin(c->prev);
        Merge(c->prev, h);
      }
    }
  }
  return coalesced_chunk;
}

// Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
// We merge Chunk(h2) into Chunk(h1).
void GPUStreamAwareBFCAllocator::Merge(BFCAllocator::ChunkHandle h1,
                                       BFCAllocator::ChunkHandle h2) {
  Chunk* c1 = ChunkFromHandle(h1);
  Chunk* c2 = ChunkFromHandle(h2);
  // We can only merge chunks that are not in use.
  CHECK(!c1->in_use() && !c2->in_use());

  int stream_1 = chunk_prev_used_streams_[h1];
  int stream_2 = chunk_prev_used_streams_[h2];

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3

  BFCAllocator::ChunkHandle h3 = c2->next;
  c1->next = h3;
  CHECK(c2->prev == h1);
  if (h3 != kInvalidChunkHandle) {
    BFCAllocator::Chunk* c3 = ChunkFromHandle(h3);
    c3->prev = h1;
  }

  // Set the new size
  c1->size += c2->size;

  // Pick latest free time.
  c1->freed_at_count = std::max(c1->freed_at_count, c2->freed_at_count);

  DEBUG_LOG << "----------\nMerge, chunk 1 size: " << c1->size
            << ", stream: " << chunk_prev_used_streams_[h1]
            << ", chunk 2 size: " << c2->size
            << ", stream: " << chunk_prev_used_streams_[h2];
  if (stream_1 >= 0 && stream_2 >= 0 && stream_1 != stream_2) {
    compute_streams_[stream_1]->ThenWaitFor(compute_streams_[stream_2]);
  }
  DeleteChunk(h2);
}

bool GPUStreamAwareBFCAllocator::Extend(size_t alignment,
                                        size_t rounded_bytes) {
  size_t available_bytes = memory_limit_ - total_region_allocated_bytes_;
  // Rounds available_bytes down to the nearest multiple of
  // kMinAllocationSize.
  available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

  // >= 0 for setting the MB limit for extend
  static const int64 extend_limit_mb = [] {
    int64 limit;
    TF_CHECK_OK(ReadInt64FromEnvVar("TF_BFC_EXTEND_LIMIT_MB",
                                         /*default_val=*/-1, &limit));
    return limit;
  }();
  static const size_t extend_limit_bytes = extend_limit_mb * (2 << 19);

  if (bfc_custom_log) {
    VLOG(0) << "Allocator: " << name_ << ", memory_limit_: "
            << strings::HumanReadableNumBytes(memory_limit_)
            << ", total_region_allocated_bytes_: "
            << strings::HumanReadableNumBytes(total_region_allocated_bytes_)
            << " ,reserve + in use: "
            << strings::HumanReadableNumBytes(stats_.bytes_reserved +
                                              stats_.bytes_in_use)
            << ", curr_region_allocation_bytes_: "
            << strings::HumanReadableNumBytes(curr_region_allocation_bytes_)
            << ", extend_limit: "
            << strings::HumanReadableNumBytes(extend_limit_bytes)
            << ", requested rounded_bytes: "
            << strings::HumanReadableNumBytes(rounded_bytes)
            << ", available_bytes: "
            << strings::HumanReadableNumBytes(available_bytes);
  }

  // Do we have enough space to handle the client's request?
  // If not, fail immediately.
  if (rounded_bytes > available_bytes) {
    return false;
  }

  // If curr_region_allocation_bytes_ is not enough to satisfy the
  // allocation, keep multiplying by a power of two until that is
  // sufficient.
  bool increased_allocation = false;
  while (rounded_bytes > curr_region_allocation_bytes_) {
    curr_region_allocation_bytes_ =
        curr_region_allocation_bytes_ <=
                std::numeric_limits<long long int>::max() / 2
            ? curr_region_allocation_bytes_ * 2
            : std::numeric_limits<long long int>::max();
    increased_allocation = true;
  }

  // Try allocating.
  size_t bytes = curr_region_allocation_bytes_;
  if (extend_limit_bytes >= 0) {
    if (extend_limit_bytes >= rounded_bytes) {
      bytes = std::min(extend_limit_bytes, curr_region_allocation_bytes_);
    } else {
      bytes = rounded_bytes;
    }
  }
  bytes = std::min(bytes, available_bytes);
  size_t bytes_received;
  void* mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
  if (mem_addr == nullptr && !started_backpedal_) {
    // Only backpedal once.
    started_backpedal_ = true;

    static constexpr float kBackpedalFactor = 0.9;

    // Try allocating less memory.
    while (mem_addr == nullptr) {
      bytes = RoundedBytes(bytes * kBackpedalFactor);
      if (bytes < rounded_bytes) break;
      mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
    }
  }

  if (mem_addr == nullptr) {
    return false;
  }

  if (!increased_allocation) {
    // Increase the region size of the next required allocation.
    curr_region_allocation_bytes_ =
        curr_region_allocation_bytes_ <=
                std::numeric_limits<long long int>::max() / 2
            ? curr_region_allocation_bytes_ * 2
            : std::numeric_limits<long long int>::max();
  }

  if (bfc_custom_log) {
    VLOG(0) << "Extending allocation by "
            << strings::HumanReadableNumBytes(bytes_received) << " bytes for "
            << Name() << ".";
  }
  VLOG(1) << "Extending allocation by "
          << strings::HumanReadableNumBytes(bytes_received) << " bytes for "
          << Name() << ".";

  total_region_allocated_bytes_ += bytes_received;
  VLOG(1) << "Total allocated bytes: "
          << strings::HumanReadableNumBytes(total_region_allocated_bytes_);

  VLOG(1) << "Allocated memory at " << mem_addr << " to "
          << static_cast<void*>(static_cast<char*>(mem_addr) + bytes_received);

  AllocationRegion* maybe_extended_region = nullptr;
  if (coalesce_regions_) {
    maybe_extended_region =
        region_manager_.AddOrExtendAllocationRegion(mem_addr, bytes_received);
  } else {
    region_manager_.AddAllocationRegion(mem_addr, bytes_received);
  }

  // Create one large chunk for the whole memory space that will
  // be chunked later.
  ChunkHandle h = AllocateChunk();
  BFCAllocator::Chunk* c = ChunkFromHandle(h);
  c->ptr = mem_addr;
  c->size = bytes_received;
  c->allocation_id = -1;
  c->prev = kInvalidChunkHandle;
  c->next = kInvalidChunkHandle;
  c->freed_at_count = 0;
  chunk_prev_used_streams_[h] = non_stream_id_;

  region_manager_.set_handle(c->ptr, h);

  // If the region was extended, then there exists a previous chunk that
  // should be linked to the new chunk.
  if (maybe_extended_region != nullptr) {
    ChunkHandle prev =
        maybe_extended_region->get_handle(maybe_extended_region->ptr());
    BFCAllocator::Chunk* prev_chunk = ChunkFromHandle(prev);
    // Find the last recorded chunk in the extended region.
    while (prev_chunk->next != kInvalidChunkHandle) {
      prev = prev_chunk->next;
      prev_chunk = ChunkFromHandle(prev);
    }
    c->prev = prev;
    prev_chunk->next = h;
  }

  // Maybe merge adjacent chunks and insert the chunk into the right bin.
  InsertFreeChunkIntoBin(TryToCoalesce(h, /*ignore_freed_at=*/false));

  return true;
}

// GPUStreamAwareBFCWrapperAllocator
void* GPUStreamAwareBFCWrapperAllocator::AllocateRaw(
    size_t unused_alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
  //AllocationAttributes temp;
  if (allocation_attr.stream_to_allocate == -1) {
    AllocationAttributes temp(allocation_attr.retry_on_failure,
                                     allocation_attr.allocation_will_be_logged,
                                     allocation_attr.freed_by_func,
                                     stream_id_);
    return stream_aware_bfc_->AllocateRaw(unused_alignment, num_bytes, std::move(temp));
  }
  return stream_aware_bfc_->AllocateRaw(unused_alignment, num_bytes,
                                        allocation_attr);
}
void GPUStreamAwareBFCWrapperAllocator::SetComputeStream(int stream_id,
                                                         se::Stream* stream) {
  stream_id_ = stream_id;
  stream_ = stream;
  stream_aware_bfc_->AddComputeStream(stream_id, stream);
}

void GPUStreamAwareBFCWrapperAllocator::DeallocateRaw(void* ptr) {
  return stream_aware_bfc_->DeallocateRaw(ptr);
}

bool GPUStreamAwareBFCWrapperAllocator::TracksAllocationSizes() const {
  return stream_aware_bfc_->TracksAllocationSizes();
}

size_t GPUStreamAwareBFCWrapperAllocator::RequestedSize(const void* ptr) const {
  return stream_aware_bfc_->RequestedSize(ptr);
}

size_t GPUStreamAwareBFCWrapperAllocator::AllocatedSize(const void* ptr) const {
  return stream_aware_bfc_->AllocatedSize(ptr);
}

int64_t GPUStreamAwareBFCWrapperAllocator::AllocationId(const void* ptr) const {
  return stream_aware_bfc_->AllocationId(ptr);
}

absl::optional<AllocatorStats> GPUStreamAwareBFCWrapperAllocator::GetStats() {
  return stream_aware_bfc_->GetStats();
}

bool GPUStreamAwareBFCWrapperAllocator::ClearStats() {
  return stream_aware_bfc_->ClearStats();
}

void GPUStreamAwareBFCWrapperAllocator::SetSafeFrontier(uint64 count) {
  return stream_aware_bfc_->SetSafeFrontier(count);
}

AllocatorMemoryType GPUStreamAwareBFCWrapperAllocator::GetMemoryType() const {
  return stream_aware_bfc_->GetMemoryType();
}

}  // namespace tensorflow
