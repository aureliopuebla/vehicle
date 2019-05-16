#include <math_constants.h>

#define THETA_N 4

extern "C" {
/**
 * Applies a 2D Complex Convolution on a real image given a square kernel and returns its magnitude response.
 * @param image The image on which to apply the convolution operation.
 * @param rows The number of rows in 'image'.
 * @param cols The number of columns in 'image'.
 * @param response The output image where the magnitude response of the convolution will be placed.
 *                 It must be of size (rows - kernel_size / 2 * 2) X (cols - kernel_size / 2 * 2).
 * @param real_kernel The real part of the square convolution kernel to apply on 'image'.
 * @param imag_kernel The imaginary part of the square convolution kernel to apply on 'image'.
 * @param kernel_size Both the number of rows and columns in 'kernel'. Should be an odd number.
 */
__global__ void complexFilter2D(unsigned char* image, int rows, int cols, float* response,
                                float* real_kernel, float* imag_kernel, int kernel_size)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = (kernel_size >> 1);
  if (image_y < offset || image_y + offset >= rows ||
      image_x < offset || image_x + offset >= cols)
    return;  // Part of the padding lost due to lack of border information.
  int image_idx = (image_y - offset) * cols + (image_x - offset), kernel_idx = 0;
  float real_response = 0.0f, imag_response = 0.0f;
  for (int i = 0; i < kernel_size; i++)
  {
    for (int j = 0; j < kernel_size; j++)
    {
      real_response += image[image_idx] * real_kernel[kernel_idx];
      imag_response += image[image_idx] * imag_kernel[kernel_idx];
      image_idx++;
      kernel_idx++;
    }
    image_idx += cols - kernel_size;
  }
  response[(image_y - offset) * (cols - (offset << 1)) + (image_x - offset)] = (
      sqrtf(real_response*real_response + imag_response*imag_response));
}

/**
 * Combines the
 * @param energies The .
 */
__global__ void combineFilteredImages(float* energies, int rows, int cols, float* combined)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = image_y * cols + image_x;
  if (image_y >= rows || image_x >= cols)
    return;  // Out of image.
  int descending_energies_arg[THETA_N];
  float temp_energies[THETA_N];
  for (int i = 0; i < THETA_N; i++)
    temp_energies[i] = energies[THETA_N * offset + i];
  for (int i = 0; i < THETA_N; i++)
  {
    int max_idx = 0;
    float max_energy = temp_energies[0];
    for (int j = 1; j < THETA_N; j++)
      if (temp_energies[j] > max_energy)
      {
        max_idx = j;
        max_energy = temp_energies[j];
      }
    descending_energies_arg[i] = max_idx;
    temp_energies[max_idx] = -1.0f;
  }
  combined[offset] = (energies[THETA_N * offset + descending_energies_arg[0]] -
                      energies[THETA_N * offset + descending_energies_arg[3]]);
}
}