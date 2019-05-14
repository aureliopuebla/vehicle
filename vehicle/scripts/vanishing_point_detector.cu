#include <math_constants.h>

#define THETA_N 4

extern "C" {
/**
 * Gets the udisp_image from the disp_image given the bins and bin_size for the histograms.
 * Assumes that a number of threads equal to 'cols' are launched.
 * @param disp_image The original disparity image.
 * @param rows The number of rows in 'disp_image'.
 * @param cols The number of cols in 'disp_image' and thus in 'udisp_image'.
 * @param udisp_image The output array for udisp_image.
 * @param bins The number of bins for 'udisp_image' aka its number of rows.
 * @bin_size The range of values each histogram bin holds.
 */
__global__ void combineFilteredImages(float* energies, float* combined)
{
  int offset = (
    (gridDim.x * blockDim.x) * (blockDim.y * blockIdx.y + threadIdx.y) +
    blockDim.x * blockIdx.x + threadIdx.x);
  for (int i = 0; i < 1; i++)
    combined[offset] = energies[THETA_N * offset + i];
  // float ordered_energies_arg[THETA_N];
}
}