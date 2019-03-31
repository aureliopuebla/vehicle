#include <curand_kernel.h>

#define THREADS 1024
#define TRIES_PER_THREAD 100
#define RANSAC_EPSILON 2

__device__ curandState_t* g_states[THREADS];

extern "C" {
__global__ void initkernels(int seed)
{
  int tid = threadIdx.x;
  curandState_t* s = new curandState_t;
  if (s != 0)
    curand_init(seed, tid, 0, s);
  g_states[tid] = s;
}

__device__ int findLowerBound(int A[], int N, int x)
{
  int l = 0;
  int h = N;
  while (l < h)
  {
    int mid = (l + h) >> 1;
    if (x <= A[mid])
      h = mid;
    else
      l = mid + 1;
  }
  return l;
}

__global__ void getVdispLine(int* vdisp_image, int rows, int cols, int* vdisp_cum_sum_array, int acc_vdisp, float* m,
                             float* b)
{
  int tid = threadIdx.x;
  curandState_t s = *g_states[tid];

  // RANSAC tries
  float best_m, best_b, best_f = 0.0;
  for (int i = 0; i < TRIES_PER_THREAD; i++)
  {
    int r1 = acc_vdisp * curand_uniform(&s) + 1;
    int r2 = acc_vdisp * curand_uniform(&s) + 1;

    int idx1 = findLowerBound(vdisp_cum_sum_array, rows * cols, r1);
    int idx2 = findLowerBound(vdisp_cum_sum_array, rows * cols, r2);

    int y1 = idx1 / cols;
    int x1 = idx1 - y1 * cols;
    int y2 = idx2 / cols;
    int x2 = idx2 - y2 * cols;
    if (x1 == x2)
      continue;  // Do not consider vertical lines
    float m = float(y2 - y1) / (x2 - x1);
    float b = y1 - m * x1;

    float f = 0.0;
    for (int x = 0; x < cols; x++)
    {
      int y = m * x + b;
      if (y < 0 || y >= rows)
        break;
      for (int yp = max(0, y - RANSAC_EPSILON); yp <= min(rows - 1, y + RANSAC_EPSILON); yp++)
        f += vdisp_image[yp * cols + x];
    }

    if (f > best_f)
    {
      best_m = m;
      best_b = b;
      best_f = f;
    }
  }

  // Reduction
  // best_lines[][0] = m, best_lines[][1] = b, best_lines[][2] = f
  __shared__ float best_lines[THREADS][3];

  best_lines[tid][0] = best_m;
  best_lines[tid][1] = best_b;
  best_lines[tid][2] = best_f;
  for (int i = THREADS >> 1; i > 0; i >>= 1)
  {
    __syncthreads();
    if (tid < i && best_lines[tid][2] < best_lines[tid + i][2])
    {
      best_lines[tid][0] = best_lines[tid + i][0];
      best_lines[tid][1] = best_lines[tid + i][1];
      best_lines[tid][2] = best_lines[tid + i][2];
    }
  }

  if (tid == 0)
  {
    *m = best_lines[0][0];
    *b = best_lines[0][1];
  }

  *g_states[tid] = s;
}
}
