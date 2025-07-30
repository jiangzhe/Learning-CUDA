#include <algorithm>
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../tester/utils.h"

// add thrust only for benchmark.
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

/// Warmup kernel for benchmark.
__global__ void warmup() {}

/// Perform block-level exclusive scan.
/// The result will be put into out vector.
/// If a sum vector is specified, each block sum
/// will be stored in it.
template <typename T>
__global__ void block_exclusive_scan(const T *g_in, int n, T *g_out, T *g_sum) {
  // shared memory, size = blockDim * 2
  extern __shared__ char s_char[];

  T *s_data = reinterpret_cast<T *>(s_char);

  int tid = threadIdx.x;
  int blockOffset = blockIdx.x * blockDim.x * 2;
  int pos = blockOffset + tid;
  // initialize
  if (pos < n) {
    s_data[tid] = g_in[pos];
  } else {
    s_data[tid] = 0;
  }
  if (pos + blockDim.x < n) {
    s_data[tid + blockDim.x] = g_in[pos + blockDim.x];
  } else {
    s_data[tid + blockDim.x] = 0;
  }
  __syncthreads();

  // up-sweep
  for (int stride = 1; stride < blockDim.x * 2; stride *= 2) {
    int idx = (threadIdx.x + 1) * 2 * stride - 1;
    if (idx < blockDim.x * 2) {
      s_data[idx] += s_data[idx - stride];
    }
    __syncthreads();
  }

  // store sum and clean last value.
  if (threadIdx.x == 0) {
    if (g_sum != nullptr) {
      g_sum[blockIdx.x] = s_data[blockDim.x * 2 - 1];
    }
    s_data[blockDim.x * 2 - 1] = 0;
  }
  __syncthreads();

  // down-sweep
  for (int stride = blockDim.x; stride > 0; stride /= 2) {
    int idx = (threadIdx.x + 1) * 2 * stride - 1;
    if (idx < blockDim.x * 2) {
      int v = s_data[idx - stride];
      s_data[idx - stride] = s_data[idx];
      s_data[idx] += v;
    }
    __syncthreads();
  }

  if (pos < n) {
    g_out[pos] = s_data[tid];
  }
  if (pos + blockDim.x < n) {
    g_out[pos + blockDim.x] = s_data[tid + blockDim.x];
  }
}

/// Add block sum to all data in corresponding block.
/// Each thread process two elements.
template <typename T>
__global__ void add_block_sums(const T *g_block_sums, T *g_out, int n) {
  int pos = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (pos < n) {
    g_out[pos] += g_block_sums[blockIdx.x];
  }
  if (pos + blockDim.x < n) {
    g_out[pos + blockDim.x] += g_block_sums[blockIdx.x];
  }
}

/// Entry of exclusive scan.
/// Input and output vectors must be on device.
/// They can be identical if inplace update is desired.
template <typename T> void exclusive_scan(T *g_in, int n, T *g_out) {
  const int threadsPerBlock = 256;
  const int elementsPerBlock = threadsPerBlock * 2;
  const int numBlocks = (n + elementsPerBlock - 1) / elementsPerBlock;
  if (numBlocks == 1) {
    size_t sharedMemSize = elementsPerBlock * sizeof(T);
    block_exclusive_scan<<<1, threadsPerBlock, sharedMemSize>>>(g_in, n, g_out,
                                                                (T *)nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    return;
  }
  // initialize block sum vector.
  T *g_block_sum;
  CUDA_CHECK(cudaMalloc(&g_block_sum, numBlocks * sizeof(T)));

  // block-level exclusive scan.
  size_t sharedMemSize = elementsPerBlock * sizeof(T);
  block_exclusive_scan<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      g_in, n, g_out, g_block_sum);
  CUDA_CHECK(cudaGetLastError());

  // exclusive scan on block sums.
  T *g_scanned_block_sum;
  CUDA_CHECK(cudaMalloc(&g_scanned_block_sum, numBlocks * sizeof(T)));
  exclusive_scan(g_block_sum, numBlocks, g_scanned_block_sum);
  CUDA_CHECK(cudaDeviceSynchronize());

  // add scanned block sum back to intermediate vector.
  add_block_sums<<<numBlocks, threadsPerBlock>>>(g_scanned_block_sum, g_out, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // free memory.
  CUDA_CHECK(cudaFree(g_scanned_block_sum));
  CUDA_CHECK(cudaFree(g_block_sum));
}

/// Inplace exclusive scan.
template <typename T> void exclusive_scan_inplace(T *g_data, int n) {
  exclusive_scan(g_data, n, g_data);
}

#define RADIX_BITS 4
#define RADIX_SLOTS 16
#define RADIX_BITS_MASK 15

/// Block-level histogram
__global__ void block_radix_histogram(const unsigned int *g_in, int n,
                                      unsigned int *g_block_histogram,
                                      int pass) {
  // 4-bit per pass, so 16 slots.
  __shared__ unsigned int s_histogram[RADIX_SLOTS];

  // initialize local histogram.
  int tid = threadIdx.x;
  if (tid < RADIX_SLOTS) {
    s_histogram[tid] = 0;
  }
  __syncthreads();

  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int v = g_in[pos];
  unsigned int radix = (v >> (pass * RADIX_BITS)) & RADIX_BITS_MASK;
  atomicAdd(&s_histogram[radix], 1);
  __syncthreads();

  if (tid < RADIX_SLOTS) {
    g_block_histogram[blockIdx.x * RADIX_SLOTS + tid] = s_histogram[tid];
  }
}

/// Transpose histogram to block-major[B, 16] or radix-major[16, B].
/// gridDim and blockDim should match original vector.
/// e.g.
/// [B, 16] => [16, B]: gridDim(1, B.div_ceil(16)), blockDim(16, 16)
/// [16, B] => [B, 16]: gridDim(B.div_ceil(16), 1), blockDim(16, 16)
__global__ void histogram_transpose(unsigned int *g_in, int width, int height,
                                    unsigned int *g_out) {
  __shared__ unsigned int tile[RADIX_SLOTS]
                              [RADIX_SLOTS + 1]; // decrease bank conflict.

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int cell_x = blockIdx.x * RADIX_SLOTS;
  int cell_y = blockIdx.y * RADIX_SLOTS;

  int g_x = cell_x + tx;
  int g_y = cell_y + ty;

  // coalesced read from global to shared memory.
  if (g_x < width && g_y < height) {
    tile[ty][tx] = g_in[g_y * width + g_x];
  }
  __syncthreads();

  // output, must make sure write can be coalesced.
  g_x = cell_y + tx;
  g_y = cell_x + ty;
  if (g_x < height && g_y < width) {
    g_out[g_y * height + g_x] = tile[tx][ty];
  }
}

#define WARP_SIZE 32

/// Reorder elements according to global offset, calculated by exclusive scan.
__global__ void radix_reorder(unsigned int *g_in, int n, unsigned int *g_out,
                              unsigned int *g_offset, int pass) {
  // 1. copy of global input: blockDim.x * sizeof(unsigned int)
  // 2. copy of global_offsets: 16 * sizeof(unsigned int)
  // 3. wrap histogram. blockDim.x / WARP_SIZE * 16 * sizeof(unsigned int)
  extern __shared__ unsigned int s_data[];

  unsigned int *s_in = s_data;
  unsigned int *s_global_offsets = &s_data[blockDim.x];
  unsigned int *s_warp_histogram = &s_data[blockDim.x + RADIX_SLOTS];

  int tid = threadIdx.x;
  int pos = blockIdx.x * blockDim.x + tid;
  int warpId = tid / WARP_SIZE;
  int numWarps = blockDim.x / WARP_SIZE;
  int laneId = tid % WARP_SIZE;

  // copy data
  if (pos < n) {
    s_in[tid] = g_in[pos];
  } else {
    s_in[tid] = 0;
  }
  __syncthreads();

  // copy offsets
  if (tid < RADIX_SLOTS) {
    s_global_offsets[tid] = g_offset[blockIdx.x * RADIX_SLOTS + tid];
  }
  __syncthreads();

  // initialize warp histogram
  if (tid < numWarps * RADIX_SLOTS) {
    s_warp_histogram[tid] = 0;
  }
  __syncthreads();

  // update warp histogram
  unsigned int v = s_in[tid];
  unsigned int radix = (v >> (pass * RADIX_BITS)) & RADIX_BITS_MASK;
  atomicAdd(&s_warp_histogram[warpId * RADIX_SLOTS + radix], 1);
  __syncthreads();

  // exclusive scan on warp histogram, using only wrap-0 threads
  if (tid < RADIX_SLOTS) {
    unsigned int ps = s_warp_histogram[tid];
    s_warp_histogram[tid] = 0;
    for (int wid = 1; wid < numWarps; wid++) {
      unsigned int v = ps;
      ps += s_warp_histogram[wid * RADIX_SLOTS + tid];
      s_warp_histogram[wid * RADIX_SLOTS + tid] = v;
    }
  }
  __syncthreads();

  // sequential scan per warp.
  if (laneId == 0) {
    for (int i = 0; i < WARP_SIZE; i++) {
      int in_idx = warpId * WARP_SIZE + i;
      unsigned int v = s_in[in_idx];
      unsigned int radix = (v >> (pass * RADIX_BITS)) & RADIX_BITS_MASK;
      unsigned int local_pos = s_warp_histogram[warpId * RADIX_SLOTS + radix];
      unsigned int final_idx = s_global_offsets[radix] + local_pos;
      s_warp_histogram[warpId * RADIX_SLOTS + radix] += 1;
      if (final_idx < n) {
        g_out[final_idx] = v;
      }
    }
  }
}

/// Prepare different data type for radix sort.
/// Radix sort requires bitwise split and order.
/// Only int and float are implemented.
template <typename T> __global__ void prepare_for_radix_sort(T *h_in, int n);

/// Flip most significant bit for int.
template <> __global__ void prepare_for_radix_sort(int *g_data, int n) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int *d_data = (unsigned int *)g_data;
  d_data[pos] ^= 0x80000000;
}

/// Flip most significant bit if non-negative float.
/// Reverse all bits if negative float.
template <> __global__ void prepare_for_radix_sort(float *g_data, int n) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int *d_data = (unsigned int *)g_data;
  unsigned int v = d_data[pos];
  if ((v & 0x80000000) == 0) {
    g_data[pos] = v ^ 0x80000000;
  } else {
    g_data[pos] = ~v;
  }
}

/// Radix sort with buffer.
/// Input will be overwritten with ascending order.
template <typename T> void radix_sort(T *g_in, int n) {
  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  prepare_for_radix_sort<<<numBlocks, threadsPerBlock>>>(g_in, n);

  // initialize buffer
  unsigned int *g_buf;
  CUDA_CHECK(cudaMalloc(&g_buf, n * sizeof(unsigned int)));

  // initialize histogram.
  size_t n_histogram = numBlocks * RADIX_SLOTS;
  unsigned int *g_block_histogram;
  CUDA_CHECK(
      cudaMalloc(&g_block_histogram, n_histogram * sizeof(unsigned int)));
  unsigned int *g_radix_histogram;
  CUDA_CHECK(
      cudaMalloc(&g_radix_histogram, n_histogram * sizeof(unsigned int)));

  dim3 transposeBlockDim(RADIX_SLOTS, RADIX_SLOTS);
  // block-major[B, 16] to radix-major[16, B]
  dim3 towardGridDim(1, (numBlocks + RADIX_SLOTS - 1) / RADIX_SLOTS);
  // radix-major[16, B] to block-major[B, 16]
  dim3 backwardGridDim((numBlocks + RADIX_SLOTS - 1) / RADIX_SLOTS, 1);

  unsigned int *d_in = reinterpret_cast<unsigned int *>(g_in);
  unsigned int *d_out = g_buf;
  int reorderSharedMem = threadsPerBlock * sizeof(unsigned int) +
                         RADIX_SLOTS * 2 * sizeof(unsigned int);

  for (int pass = 0; pass < 32 / RADIX_BITS; pass++) {
    // calculate histogram.
    block_radix_histogram<<<numBlocks, threadsPerBlock>>>(
        d_in, n, g_block_histogram, pass);
    CUDA_CHECK(cudaDeviceSynchronize());
    // transpose to radix-major.
    histogram_transpose<<<towardGridDim, transposeBlockDim>>>(
        g_block_histogram, RADIX_SLOTS, numBlocks, g_radix_histogram);
    CUDA_CHECK(cudaDeviceSynchronize());
    // exclusive scan.
    exclusive_scan_inplace(g_radix_histogram, n_histogram);
    CUDA_CHECK(cudaDeviceSynchronize());
    // transpose back to block-major.
    histogram_transpose<<<backwardGridDim, transposeBlockDim>>>(
        g_radix_histogram, numBlocks, RADIX_SLOTS, g_block_histogram);
    CUDA_CHECK(cudaDeviceSynchronize());
    // reorder
    radix_reorder<<<numBlocks, threadsPerBlock, reorderSharedMem>>>(
        d_in, n, d_out, g_block_histogram, pass);
    CUDA_CHECK(cudaDeviceSynchronize());
    // swap buffer pointers.
    std::swap(d_in, d_out);
  }
  // after 8 iterations, output is just in g_in, so no need to copy data.

  // clean up memory.
  CUDA_CHECK(cudaFree(g_radix_histogram));
  CUDA_CHECK(cudaFree(g_block_histogram));
  CUDA_CHECK(cudaFree(g_buf));
}

/// Convert result back from unsigned int.
template <typename T> T from_uint(unsigned int v);

template <> int from_uint(unsigned int v) {
  union {
    unsigned int uv;
    int iv;
  } conv;
  conv.uv = v ^ 0x80000000;
  return conv.iv;
}

template <> float from_uint(unsigned int v) {
  union {
    unsigned int uv;
    float fv;
  } conv;
  if ((v & 0x80000000) == 0) {
    conv.uv = ~v;
  } else {
    conv.uv = v ^ 0x8000000;
  }
  return conv.fv;
}

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 *
 * @tparam T Type of elements in the input vector (should support `int` and
 `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the
 largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant
 CPU allowed.
 * @note Library functions that can directly complete a significant part of the
 work are NOT allowed.
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally.
 Errors should be thrown.
 */
template <typename T> T kthLargest(const std::vector<T> &h_input, size_t k) {
  size_t size = h_input.size();
  if (size < k) {
    return T(-100);
  }

  // initialize input vector on GPU global memory.
  T *d_input;
  CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(T)));
  // copy input to GPU.
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(T),
                        cudaMemcpyHostToDevice));

  // radix sort.
  radix_sort(d_input, size);

  // fetch result.
  unsigned int u;
  CUDA_CHECK(
      cudaMemcpy(&u, &d_input[size - k], sizeof(int), cudaMemcpyDeviceToHost));
  T res = from_uint<T>(u);
  // clean up.
  CUDA_CHECK(cudaFree(d_input));

  return res;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads,
 * head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len,
 * query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query
 * attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  // flashAttention algorithm as below:
  //
  // for k = 1 to N do
  //
  //   for i = 1 to N do
  //     x[i] = Q[k,:] @ (K.T)[:,i]
  //     m[i] = max(m[i-1], x[i])
  //     d[i] = d[i-1] * exp(m[i-1]-m[i]) + exp(x[i]-m[i])
  //     o[i] = o[i-1] * d[i-1] * exp(m[i-1] - m[i]) / d[i] + \
  //       exp(x[i] - m[i]) / d[i] * V[i,:]
  //
  //     O[k,:] = o[N]
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int> &, size_t);
template float kthLargest<float>(const std::vector<float> &, size_t);
template void flashAttention<float>(const std::vector<float> &,
                                    const std::vector<float> &,
                                    const std::vector<float> &,
                                    std::vector<float> &, int, int, int, int,
                                    int, int, bool);

constexpr int SIZE = 1024 * 1024;

int main(void) {

  std::vector<int> h_data;
  srand(static_cast<unsigned int>(time(0)));
  int k = 1000;
  for (int i = 0; i < SIZE; i++) {
    h_data.push_back(rand() % 1000000);
  }

  thrust::host_vector<int> h_vec(SIZE);
  for (int i = 0; i < SIZE; i++) {
    h_vec[i] = h_data[i];
  }

  // warm up
  warmup<<<1, 1>>>();
  CUDA_CHECK(cudaGetLastError());

  // benchmark
  auto start = std::chrono::high_resolution_clock::now();
  auto res = kthLargest(h_data, k);
  auto end = std::chrono::high_resolution_clock::now();
  auto dur1 = end - start;

  start = std::chrono::high_resolution_clock::now();
  std::sort(h_data.begin(), h_data.end());
  end = std::chrono::high_resolution_clock::now();
  auto dur2 = end - start;

  start = std::chrono::high_resolution_clock::now();
  thrust::device_vector<int> d_vec = h_vec;
  thrust::sort(d_vec.begin(), d_vec.end());
  h_vec = d_vec;
  end = std::chrono::high_resolution_clock::now();
  auto dur3 = end - start;

  std::cout
      << "res on gpu is " << res << ", elapsed time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(dur1).count()
      << " microsec" << std::endl;
  std::cout
      << "res on cpu is " << h_data[h_data.size() - k] << ", elapsed time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(dur2).count()
      << " microsec" << std::endl;
  std::cout
      << "res with thrust is " << h_vec[h_data.size() - k] << ", elapsed time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(dur3).count()
      << " microsec" << std::endl;
  return 0;
}
