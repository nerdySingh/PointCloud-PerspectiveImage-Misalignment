#include "PointCloudTransformer.h"
#include "math_functions.h"
#include "device_atomic_functions.h"

#define GPU_WARP_SIZE 32

__global__ void FloatGPUMemset_GPUKernel(float *d_array,
                                         int array_size, float val) {
  int idx = ((gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x) 
    + threadIdx.x;
  d_array[idx % array_size] = val;
}

__global__ void LLA2NEmU_GPUKernel(float *h_lla_data, float *d_out_coord_data,
                                  float eq_radius, float ecc, int num_points,
                                  float x0, float y0, float z0) {
  int phi_idx = ((gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x);

  if (phi_idx < num_points * 3) {
    int lambda_idx = phi_idx + 1;
    int h_idx = phi_idx + 2;

    float phi = h_lla_data[phi_idx];
    float lambda = h_lla_data[lambda_idx];
    float h = h_lla_data[h_idx];

    float sin_phi = sinf(phi);
    float cos_phi = cosf(phi);
    float sin_lambda = sinf(lambda);
    float cos_lambda = cosf(lambda);

    float N_phi = eq_radius / sqrtf(1 - powf(ecc * sin_phi, 2.0f));
    float xy_p0 = (h + N_phi) * cos_lambda;

    if (threadIdx.x == 0)
      d_out_coord_data[phi_idx] = (xy_p0 * cos_phi) - x0;
    else if (threadIdx.x == 1)
      d_out_coord_data[lambda_idx] = (xy_p0 * sin_phi) - y0;
    else
      d_out_coord_data[h_idx] = ((h + (1 - ecc * ecc) * N_phi) * sin_lambda)
                                - z0;
    __syncthreads();

    float x = d_out_coord_data[phi_idx];
    float y = d_out_coord_data[lambda_idx];
    float z = d_out_coord_data[h_idx];

    __syncthreads();

    if (threadIdx.x == 0)
      d_out_coord_data[lambda_idx] = (x * (-sin_lambda)) + (y * cos_lambda);
    else if (threadIdx.x == 1)
      d_out_coord_data[phi_idx] = (x * (-cos_lambda) * sin_phi) + (y * (-sin_phi)
         * sin_lambda) + (z * cos_phi);
    else
      d_out_coord_data[h_idx] = -((x * cos_phi
                                   * cos_lambda)
                                  + (y * cos_phi
                                     * sin_lambda)
                                  + (z * sin_phi));

    //if (threadIdx.x == 0)
    //  d_out_coord_data[phi_idx] = (x * (-sin_lambda)) + (y * cos_lambda);
    //else if (threadIdx.x == 1)
    //  d_out_coord_data[lambda_idx] = (x * (-cos_lambda) * sin_phi) + (y * (-sin_phi)
    //                                                               * sin_lambda) + (z * cos_phi);
    //else
    //  d_out_coord_data[h_idx] = ((x * cos_phi
    //                               * cos_lambda)
    //                              + (y * cos_phi
    //                                 * sin_lambda)
    //                              + (z * sin_phi));
  }
}



__global__ void CamCoord2Img_GPUKernel(float *d_cam_xyz_data, 
                                       float *h_intensities,
                                       float *d_front_img, float *d_rear_img,
                                       float *d_left_img, float *d_right_img,
                                       int resolution, int num_samples) {
  int point_idx = ((gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x)
    + threadIdx.x;

  if (point_idx < num_samples) {
    int cam_x_idx = point_idx * 3;
    float x = d_cam_xyz_data[cam_x_idx];
    float y = d_cam_xyz_data[cam_x_idx + 1];
    float z = d_cam_xyz_data[cam_x_idx + 2];

    int projected_x, projected_y;

    if (z < 0 && z < -abs(x) && z < -abs(y)) { //REAR
      projected_x = (int)(((float)(-y / z) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;
      projected_y = (int)(((float)(-x / z) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;

      int projected_idx = projected_y * resolution + projected_x;
      //d_front_img[1] = projected_idx;
      d_rear_img[projected_idx] = 255;
      //atomicAdd(&d_front_img[projected_idx], h_intensities[point_idx]);
    }
    else if (z > 0 && z > abs(x) && z > abs(y)) { //FRONT
      projected_x = (int)(((float)(y / z) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;
      projected_y = (int)(((float)(x / z) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;

      int projected_idx = projected_y * resolution + projected_x;
      //d_front_img[1] = projected_idx;
      d_front_img[projected_idx] = 255;
      //atomicAdd(&d_front_img[projected_idx], h_intensities[point_idx]);
    }

  }
}

void FloatGPUMemset(float *d_array, int array_size, float val) {
  int threadBlock_size = GPU_WARP_SIZE * 2;
  int num_threadblocks = std::ceilf((float)array_size / threadBlock_size);
  int tb_rows = (int)std::sqrtf((float)num_threadblocks);
  int tb_cols = (int)std::ceilf((float)num_threadblocks / tb_rows);
  dim3 threadBlock_dim(tb_rows, tb_cols, 1);
  FloatGPUMemset_GPUKernel <<< threadBlock_dim, threadBlock_size >>> 
    (d_array, array_size, val);
}

void LLA2NEmU_GPU(float *h_lla_data, float *d_nemu_data, float eq_radius,
                 float ecc, int num_samples, int sample_size,
                 cam_details &ref_cam) {
  int tb_rows = (int)std::sqrtf((float)num_samples);
  int tb_cols = (int)std::ceilf((float)num_samples / tb_rows);
  dim3 threadBlock_dim(tb_rows, tb_cols, 1);
  LLA2NEmU_GPUKernel <<< threadBlock_dim, sample_size >>> (h_lla_data,
                                                           d_nemu_data,
                                                           eq_radius, ecc,
                                                           num_samples,
                                                           ref_cam.x,
                                                           ref_cam.y,
                                                           ref_cam.z);
}

void CamCoord2Img_GPU(float *d_cam_xyz_data, float *h_intensities,
                      float *d_front_img, float *d_rear_img,
                      float *d_left_img, float *d_right_img,
                      int resolution, int num_samples) {
  int threadBlock_size = GPU_WARP_SIZE * 2;
  int threads = (int)std::ceilf((float)num_samples / threadBlock_size);
  int blocks = (int)std::ceilf((float)threads / threadBlock_size);
  int grid_rows = (int)std::sqrtf((float)blocks);
  int grid_cols = (int)std::ceilf((float)blocks / grid_rows);
  dim3 threadBlock_dim(grid_rows, grid_cols, 1);
  CamCoord2Img_GPUKernel <<< threadBlock_dim, threadBlock_size >>> 
    (d_cam_xyz_data, h_intensities, d_front_img, d_rear_img,
     d_left_img, d_right_img, resolution, num_samples);
}