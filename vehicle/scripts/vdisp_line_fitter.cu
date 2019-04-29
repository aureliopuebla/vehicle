#include <curand_kernel.h>

#define THREADS ${threads}
#define TRIES_PER_THREAD ${tries_per_thread}
#define RANSAC_EPSILON ${ransac_epsilon}
#define RANSAC_EPSILON_DECAY ${ransac_epsilon_decay}

__device__ curandState_t* g_states[THREADS];

extern "C" {
/**
 * Initializes the random states for all threads that will execute RANSAC tries.
 * @param seed The random seed to use. It's recommended setting it to unix time.
 */
__global__ void initRandomStates(int seed)
{
  int tid = threadIdx.x;
  curandState_t* s = new curandState_t;
  if (s != 0)
    curand_init(seed, tid, 0, s);
  g_states[tid] = s;
}

/** TODO: OPTIMIZE TYPES **/

/**
 * Gets the udisp_image from the disp_image given the bins and bin_size for the histograms.
 * It assumes that a number of threads equal to 'cols' are launched.
 * @param disp_image The original disparity image.
 * @param rows The number of rows in 'disp_image'.
 * @param cols The number of cols in 'disp_image' and thus in 'udisp_image'.
 * @param udisp_image The output array for udisp_image.
 * @param bins The number of bins for 'udisp_image' aka its number of rows.
 * @bin_size The range of values each histogram bin holds.
 */
__global__ void getUDisparity(int* disp_image, int rows, int cols, int* udisp_image, int bins, int bin_size)
{
  int tid = threadIdx.x;
  int disp_image_idx = tid;
  for (int i = 0; i < rows; i++)
  {
    int bin = disp_image[disp_image_idx] / bin_size;
    if(bin < bins)
      udisp_image[bin*cols + tid]++;
    disp_image_idx += cols;
  }
}

/**
 * Helper function that does binary search on a value returning the lower bound.
 * @param A The sorted array in which to search.
 * @param N The size of array 'A'.
 * @param x The element to search for.
 * @return The lower bound index in 'A' after the search.
 */
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

/**
 * Executes and reduces RANSAC tries to get a fitted line on the given vdisp.
 * @param vdisp_image The filtered vdisp image on which to fit the line.
 * @param rows The number of rows in 'vdisp_image'.
 * @param cols The number of columns in 'vdisp_image'.
 * @param vdisp_cum_sum_array The cumulative sum in a linearized version of vdisp_image.
 * @param acc_disp The maximum value in 'vdisp_cum_sum_array '.
 * @param m The address where the resulting vdisp_line slope will be returned.
 * @param b The address where the resulting vdisp_line row-intersect will be returned.
 */
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

    float fs[RANSAC_EPSILON + 1];
    for (int yp = 0; yp <= RANSAC_EPSILON; yp++)
        fs[yp] = 0.0;
    for (int x = 0; x < cols; x++)
    {
      int y = m * x + b;
      if (y < 0 || y >= rows)
        break;
      for (int yp = max(0, y - RANSAC_EPSILON); yp <= min(rows - 1, y + RANSAC_EPSILON); yp++)
        fs[abs(yp - y)] += vdisp_image[yp * cols + x];
    }
    float f = 0.0;
    for (int yp = 0; yp <= RANSAC_EPSILON; yp++)
        f += fs[yp] * pow(1 - RANSAC_EPSILON_DECAY, yp);

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
