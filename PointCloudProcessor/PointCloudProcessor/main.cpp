#include <iostream>

#include "PointCloudTransformer.h"

using namespace std;

#define NUM_POINTS 430736

#define NUM_POINTS2 1046233

void print_h_var(float *h_v, int r, int c, bool print_elem = true) {
  std::cout << "-------------------------" << std::endl;
  float mini = h_v[0], maxi = h_v[0];
  float sum = 0.0f;
  int mini_idx = 0, maxi_idx = 0;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      if (print_elem)
        //std::cout << h_v[j + i * c] << "\t";
        printf("%.9f\t", h_v[j + i * c]);
      if (h_v[j + i * c] < mini) {
        mini = h_v[j + i * c];
        mini_idx = j + i * c;
      }
      if (h_v[j + i * c] > maxi) {
        maxi = h_v[j + i * c];
        maxi_idx = j + i * c;
      }
      sum += h_v[j + i * c];
    }
    if (print_elem)
      std::cout << std::endl;
  }
  std::cout << "Shape = (" << r << ", " << c << ")" << std::endl;
  std::cout << "Minimum at index " << mini_idx << " = " << mini << std::endl;
  std::cout << "Maximum at index " << maxi_idx << " = " << maxi << std::endl;
  std::cout << "Average of all elements = " << sum / (r * c) << std::endl;
  // std::cout << std::endl;
}

void print_d_var(float *d_v, int r, int c, bool print_elem = true) {
  std::cout << "*****************************" << std::endl;
  float *h_v = (float *)malloc(sizeof(float) * r * c);
  cudaMemcpy(h_v, d_v, sizeof(float) * r * c, cudaMemcpyDeviceToHost);
  float mini = h_v[0], maxi = h_v[0];
  int mini_idx = 0, maxi_idx = 0;
  float sum = 0.0;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      if (print_elem)
        printf("%f\t", h_v[j + i * c]);
      if (h_v[j + i * c] < mini) {
        mini = h_v[j + i * c];
        mini_idx = j + i * c;
      }
      if (h_v[j + i * c] > maxi) {
        maxi = h_v[j + i * c];
        maxi_idx = j + i * c;
      }
      sum += h_v[j + i * c];
    }
    if (print_elem)
      std::cout << std::endl;
  }
  std::cout << "Shape = (" << r << ", " << c << ")" << std::endl;
  std::cout << "Minimum at index " << mini_idx << " = " << mini << std::endl;
  std::cout << "Maximum at index " << maxi_idx << " = " << maxi << std::endl;
  std::cout << "Average of all elements = " << sum / (r * c) << std::endl;
  // std::cout << std::endl;
  free(h_v);
}

int main() {
  PointCloudTransformer *pcl = new PointCloudTransformer("final_project_point_cloud.fuse",
                                                         NUM_POINTS);

  pcl->LoadCameraDetails(45.90414414f, 11.02845385f, 227.5819f,
                         0.362114f, 0.374050f, 0.592222f, 0.615007f);
  //pcl->LoadCameraDetails(45.90414414f, 11.02845385f, 240.5819f,
  //                       -0.18f, 0.374050f, 0.592222f, 0.615007f);
  //pcl->LoadCameraDetails(45.90414414f, 11.02845385f, 240.5819f,
  //                       -1.0f, 0.86, -0.23, -0.1);
  print_d_var(pcl->d_Rq, 3, 3);
  pcl->PopulateReadBuffer();

  pcl->ConvertLLA2NEmU_GPU();
  pcl->ConvertNEmU2CamCoord_GPU();

  int res = 256;
  for (int i = 0; i < 4; i++) {
    pcl->ConvertCamCoord2Img_CPU(res); // outputs image of desired resolution
    res *= 2;
  }

  pcl->LoadResults();
  ofstream results_file;
  results_file.open("cam_coords.fuse", std::ofstream::out | std::ofstream::app);
  for (int i = 0; i < NUM_POINTS; i++) {
    results_file << pcl->h_positions_buffer_ptr[i * 3] << " "
      << pcl->h_positions_buffer_ptr[i * 3 + 1] << " "
      << pcl->h_positions_buffer_ptr[i * 3 + 2] << " "
      << pcl->h_intensities_buffer_ptr[i] << std::endl;
  }
  results_file.close();

  return 0;
}