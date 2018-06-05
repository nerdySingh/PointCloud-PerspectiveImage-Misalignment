#pragma once

#include <cuda.h>
#include <cublas.h>
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>

#include "cuda_err_check.h"

#define EQUATORIAL_RADIUS 6378137.0f // 'a'
#define POLAR_RADIUS 6356752.3142f // 'b'
#define POSITION_DIM 3


using namespace std;
using namespace cv;

struct cam_details {
  float phi; //lat
  float lambda; //long
  float h; //alt
  float x;
  float y;
  float z;
  float Qs;
  float Qx;
  float Qy;
  float Qz;
};

void LLA2NEmU_GPU(float *h_lla_data, float *d_nemu_data, float eq_radius,
                 float ecc, int num_samples, int sample_size,
                 cam_details &ref_cam);

void CamCoord2Img_GPU(float *d_cam_xyz_data, float *h_intensities,
                      float *d_front_img, float *d_rear_img,
                      float *d_left_img, float *d_right_img,
                      int resolution, int num_samples);

void FloatGPUMemset(float *d_array, int array_size, float val);

class PointCloudTransformer {
public:
  bool img_allocated;
  const char *filename;
  std::ifstream pointcloud_fstream;
  std::string read_line;
  //std::vector<std::vector<float>> pointcloud_buffer;
  float *h_positions_buffer_ptr, *d_positions_buffer_ptr;
  float *h_intensities_buffer_ptr;
  float *h_Rq, *d_Rq;
  
  float *d_front, *d_rear, *d_left, *d_right;
  float *h_front, *h_rear, *h_left, *h_right;

  cudaDeviceProp cudaProp;
  cublasHandle_t cublas_handle;

  int global_row_idx;
  int local_row_idx;
  int buffer_size;
  bool end_reached;

  float ellipsoidal_flattening; // 'f'
  float eccentricity; // 'e'

  cam_details reference_cam;

  int read_rows;

  PointCloudTransformer(const char *filename_arg, int buff_size = 5);
  void PopulateReadBuffer();
  void LLA2ECEF_CPU(float phi, float lambda, float h, float eq_radius,
                    float ecc, float *x, float *y, float *z);
  void LoadCameraDetails(float cam_phi, float cam_lambda, float cam_h,
                         float cam_Qs, float cam_Qx,
                         float cam_Qy, float cam_Qz);
  
  void ConvertLLA2NEmU_GPU();
  void ConvertNEmU2CamCoord_GPU();
  void ConvertCamCoord2Img_CPU(int resolution);
  void ConvertCamCoord2Img_GPU(int resolution);
  
  void NormalizeMatrix_CPU(float *h_mat, int total_size, int scale=255);
  void LoadResults();
  std::vector<float> static split(const std::string &s,
                                  char delim);

private:
  bool ReadNextRow();
  void init_Rq(float *h_Rq);
  void static split_(const std::string &s, char delim,
                     std::vector<float> &elems);

  ~PointCloudTransformer();
};

