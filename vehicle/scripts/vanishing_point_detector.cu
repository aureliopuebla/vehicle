#include <math_constants.h>

#define THETA_N 4

extern "C" {
/**
 * Clears out the Gabor Energies Tensor, setting all of its values to zero.
 * The Gabor Energies Tensor is the data structure whose [y, x, theta] value contains the average magnitude response to
 * the different complex 'Gabor' filters for an specific 'theta' orientation at 'image' location (y, x).
 * This is the first step towards its calculation. Note that the number of rows and columns in the Gabor Energies Tensor
 * is (image_rows - (kernel_size >> 1)) X (image_cols - (kernel_size >> 1)) due to the padding lost at convolution.
 * @param gabor_energies The Gabor Energies Tensor.
 * @param rows The number of rows in the 'image' whose Energies Tensor will be calculated.
 * @param cols The number of columns in the 'image' whose Energies Tensor will be calculated.
 * @param kernel_size Both the number of rows and columns in the Gabor kernels to apply. Should be an odd number.
 */
__global__ void resetGaborEnergiesTensor(float* gabor_energies, int rows, int cols, int kernel_size)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  int image_padding = (kernel_size >> 1);
  if (image_y < image_padding || image_y + image_padding >= rows ||
      image_x < image_padding || image_x + image_padding >= cols)
    return;  // Part of the padding lost due to lack of border information.
  int tensor_offset = ((image_y - image_padding) * (cols - (image_padding << 1)) + (image_x - image_padding)) * THETA_N;
  for (int i = 0; i < THETA_N; i++)
    gabor_energies[tensor_offset + i] = 0.0f;
}

/**
 * Applies a 2D Complex Convolution on a real image given a square kernel and adds its magnitude response to the
 * corresponding [y, x, theta] location in the Gabor Energies Tensor.
 * This kernel is the second step towards its calculation and should be called once for every frequency to apply.
 * @param gabor_energies The Gabor Energies Tensor.
 * @param theta_idx The orientation index for the Gabor Energies Tensor specifying the orientation for which to add
 *                  this convolution.
 * @param image The image on which to apply the convolution operation.
 * @param rows The number of rows in 'image'.
 * @param cols The number of columns in 'image'.
 * @param real_kernel The real part of the square convolution kernel to apply on 'image'.
 * @param imag_kernel The imaginary part of the square convolution kernel to apply on 'image'.
 * @param kernel_size Both the number of rows and columns in 'kernel'. Should be an odd number.
 */
__global__ void addGaborFilterMagnitudeResponse(float* gabor_energies, int theta_idx,
                                                unsigned char* image, int rows, int cols,
                                                float* real_kernel, float* imag_kernel, int kernel_size)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  int image_padding = (kernel_size >> 1);
  if (image_y < image_padding || image_y + image_padding >= rows ||
      image_x < image_padding || image_x + image_padding >= cols)
    return;  // Part of the padding lost due to lack of border information.
  int image_idx = (image_y - image_padding) * cols + (image_x - image_padding), kernel_idx = 0;
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
  int tensor_offset = ((image_y - image_padding) * (cols - (image_padding << 1)) + (image_x - image_padding)) * THETA_N;
  gabor_energies[tensor_offset + theta_idx] = sqrtf(real_response * real_response + imag_response * imag_response);
}

/**
 * Divides all of the Gabor Energies Tensor elements by a constant.
 * This is the third and last step to calculate the Tensor. This step is used to average out the magnitude responses of
 * the different applied Gabor kernels: for a given [y, x, theta], one is applied per frequency.
 * @param gabor_energies The Gabor Energies Tensor.
 * @param rows The number of rows in the 'image' whose Energies Tensor will be calculated.
 * @param cols The number of columns in the 'image' whose Energies Tensor will be calculated.
 * @param kernel_size Both the number of rows and columns in the applied Gabor kernels. Should be an odd number.
 * @param constant The number by which to divide all of the Gabor Energies Tensor elements. Should be equal to the
 *                 number of applied frequencies.
 */
__global__ void divideGaborEnergiesTensor(float* gabor_energies, int rows, int cols, int kernel_size, int constant)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  int image_padding = (kernel_size >> 1);
  if (image_y < image_padding || image_y + image_padding >= rows ||
      image_x < image_padding || image_x + image_padding >= cols)
    return;  // Part of the padding lost due to lack of border information.
  int tensor_offset = ((image_y - image_padding) * (cols - (image_padding << 1)) + (image_x - image_padding)) * THETA_N;
  for (int i = 0; i < THETA_N; i++)
    gabor_energies[tensor_offset + i] /= constant;
}

/**
 * Combines the TODO: documment this & improve kernel.
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