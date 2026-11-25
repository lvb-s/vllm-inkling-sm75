// Push-based two-stage fp16 all-reduce for four PCIe-only GPUs.
//
// The fork's cross_device_reduce_2stage pulls peers' inputs with remote loads;
// on the Broadwell root complex those non-posted reads are slower than posted
// writes. Here every rank writes the shards of its input into their owners'
// receive slots, the owners reduce them in the same order as the fork kernel
// (owner first, then rank + 1, ...) and write the sum into every rank's gather
// slots, so results are bitwise identical to the fork kernel.
//
// Buffer reuse needs no double buffering: a rank starts call n + 1 only after
// its call n has finished, which requires every peer to have passed the second
// barrier of call n (done reading its receive slots), and it writes peers'
// gather slots only after the first barrier of call n + 1 (peers done with call
// n). Both barriers are per block and wait with >= on monotonic counters, so
// calls of different sizes (grids) may follow each other.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kRanks = 4;
constexpr int kMaxBlocks = 36;
constexpr int kThreads = 128;
constexpr int64_t kSignalBytes = 4096;

struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][kRanks];
  alignas(128) uint32_t end[kMaxBlocks][kRanks];
  alignas(128) uint32_t flag[kMaxBlocks];
};
static_assert(sizeof(Signal) <= kSignalBytes);

struct Peers {
  char* p[kRanks];
};

__device__ __forceinline__ void st_release(uint32_t* addr, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(v), "l"(addr) : "memory");
}

__device__ __forceinline__ uint32_t ld_acquire(uint32_t* addr) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(addr) : "memory");
  return v;
}

template <bool kEnd>
__device__ __forceinline__ void barrier(const Peers& base, Signal* self, int rank,
                                        uint32_t f) {
  __syncthreads();
  if (threadIdx.x < kRanks) {
    auto* peer = reinterpret_cast<Signal*>(base.p[threadIdx.x]);
    uint32_t* dst = kEnd ? &peer->end[blockIdx.x][rank] : &peer->start[blockIdx.x][rank];
    uint32_t* src = kEnd ? &self->end[blockIdx.x][threadIdx.x]
                         : &self->start[blockIdx.x][threadIdx.x];
    st_release(dst, f);
    while (static_cast<int>(ld_acquire(src) - f) < 0) {
    }
  }
  __syncthreads();
}

template <bool kFirst>
__device__ __forceinline__ void add(float (&acc)[8], uint4 v) {
  const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f = __half22float2(h[i]);
    acc[2 * i] = kFirst ? f.x : acc[2 * i] + f.x;
    acc[2 * i + 1] = kFirst ? f.y : acc[2 * i + 1] + f.y;
  }
}

__global__ void __launch_bounds__(kThreads)
    push_allreduce_kernel(const uint4* __restrict__ in, uint4* __restrict__ out,
                          Peers base, int rank, int part) {
  auto* self = reinterpret_cast<Signal*>(base.p[rank]);
  const uint32_t f = self->flag[blockIdx.x] + 1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  // Per rank after the signals: receive slots [kRanks][part], gather slots [kRanks][part].
  auto slots = [&](int r) { return reinterpret_cast<uint4*>(base.p[r] + kSignalBytes); };

  for (int idx = tid; idx < part; idx += stride) {
#pragma unroll
    for (int i = 1; i < kRanks; i++) {
      int dst = (rank + i) % kRanks;
      slots(dst)[rank * part + idx] = in[dst * part + idx];
    }
  }
  barrier<false>(base, self, rank, f);

  const uint4* recv = slots(rank);
  for (int idx = tid; idx < part; idx += stride) {
    float acc[8];
    add<true>(acc, in[rank * part + idx]);
#pragma unroll
    for (int i = 1; i < kRanks; i++) add<false>(acc, recv[((rank + i) % kRanks) * part + idx]);
    uint4 sum;
    half2* h = reinterpret_cast<half2*>(&sum);
#pragma unroll
    for (int i = 0; i < 4; i++)
      h[i] = __halves2half2(__float2half(acc[2 * i]), __float2half(acc[2 * i + 1]));
    out[rank * part + idx] = sum;
#pragma unroll
    for (int i = 1; i < kRanks; i++) {
      int dst = (rank + i) % kRanks;
      slots(dst)[(kRanks + rank) * part + idx] = sum;
    }
  }
  barrier<true>(base, self, rank, f);
  if (threadIdx.x == 0) self->flag[blockIdx.x] = f;

  const uint4* gath = recv + kRanks * part;
  for (int idx = tid; idx < part; idx += stride) {
#pragma unroll
    for (int i = 1; i < kRanks; i++) {
      int src = (rank + i) % kRanks;
      out[src * part + idx] = gath[src * part + idx];
    }
  }
}

}  // namespace

void push_allreduce(const torch::Tensor& inp, torch::Tensor& out,
                    const std::vector<int64_t>& buffers, int64_t rank) {
  TORCH_CHECK(inp.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf);
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous() && inp.numel() == out.numel());
  TORCH_CHECK(buffers.size() == kRanks && rank >= 0 && rank < kRanks);
  const int64_t bytes = inp.numel() * inp.element_size();
  TORCH_CHECK(bytes % (16 * kRanks) == 0);
  const int part = static_cast<int>(bytes / 16 / kRanks);
  Peers base;
  for (int i = 0; i < kRanks; i++) base.p[i] = reinterpret_cast<char*>(buffers[i]);
  const int blocks = std::min(kMaxBlocks, (part + kThreads - 1) / kThreads);
  const at::cuda::CUDAGuard guard(inp.device());
  push_allreduce_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(inp.data_ptr()), reinterpret_cast<uint4*>(out.data_ptr()),
      base, static_cast<int>(rank), part);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("push_allreduce", &push_allreduce);
  m.attr("SIGNAL_BYTES") = kSignalBytes;
}
