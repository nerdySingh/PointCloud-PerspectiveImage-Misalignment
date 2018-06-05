#include "PointCloudTransformer.h"

void show_img(cv::Mat &img) {
  cv::Mat img_scaled = cv::Mat(600, 600, CV_8UC1);
  cv::resize(img, img_scaled, img_scaled.size());
  cv::namedWindow("image");
  cv::imshow("image", img_scaled);
  cv::waitKey();
}

void show_img_org(cv::Mat &img) {
  cv::namedWindow("image");
  cv::imshow("image", img);
  cv::waitKey();
}

void print_d_var2(float *d_v, int r, int c, bool print_elem = true) {
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

PointCloudTransformer::PointCloudTransformer(const char *filename_arg,
                                             int buff_size) {
  filename = filename_arg;
  pointcloud_fstream.open(filename);
  global_row_idx = 0;
  local_row_idx = 0;
  read_rows = 0;
  buffer_size = buff_size;
  end_reached = false;
  img_allocated = false;
  ellipsoidal_flattening = (EQUATORIAL_RADIUS - POLAR_RADIUS)
    / EQUATORIAL_RADIUS;
  eccentricity = std::sqrtf(ellipsoidal_flattening
                            * (2 - ellipsoidal_flattening));
  CudaSafeCall(cudaGetDeviceProperties(&cudaProp, 0));
  CudaSafeCall(cudaMallocHost((void **)&h_positions_buffer_ptr,
                              sizeof(float) * POSITION_DIM * buff_size));
  CudaSafeCall(cudaMalloc((void **)&d_positions_buffer_ptr,
                          sizeof(float) * POSITION_DIM * buff_size));
  CudaSafeCall(cudaMallocHost((void **)&h_intensities_buffer_ptr,
                              sizeof(float) * buff_size));
  CublasSafeCall(cublasCreate_v2(&cublas_handle));
}

void PointCloudTransformer::PopulateReadBuffer() {
  int i;
  for (i = 0; i < buffer_size; i++) {
    if (!ReadNextRow()) {
      buffer_size = i;
      end_reached = true;
      break;
    }
  }
  read_rows = i;
}

void PointCloudTransformer::LLA2ECEF_CPU(float phi, float lambda, float h, float eq_radius,
                                         float ecc, float *x, float *y, float *z) {
  float sin_phi = sinf(phi);
  float cos_phi = cosf(phi);
  float sin_lambda = sinf(lambda);
  float cos_lambda = cosf(lambda);

  float N_phi = eq_radius / sqrtf(1 - powf(ecc * sin_phi, 2.0f));
  float xy_p0 = (h + N_phi) * cos_lambda;

  *x = xy_p0 * cos_phi;
  *y = xy_p0 * sin_phi;
  *z = (h + (1 - ecc * ecc) * N_phi) * sin_lambda;
}

void PointCloudTransformer::LoadCameraDetails(float cam_phi, float cam_lambda, float cam_h,
                                              float cam_Qs, float cam_Qx,
                                              float cam_Qy, float cam_Qz) {
  reference_cam.phi = cam_phi;
  reference_cam.lambda = cam_lambda;
  reference_cam.h = cam_h;
  reference_cam.Qs = cam_Qs;
  reference_cam.Qx = cam_Qx;
  reference_cam.Qy = cam_Qy;
  reference_cam.Qz = cam_Qz;
  LLA2ECEF_CPU(reference_cam.phi, reference_cam.lambda, reference_cam.h,
               EQUATORIAL_RADIUS, eccentricity, &reference_cam.x,
               &reference_cam.y, &reference_cam.z);
  h_Rq = (float *)malloc(sizeof(float) * 3 * 3);
  init_Rq(h_Rq);
  CudaSafeCall(cudaMalloc((void **)&d_Rq, sizeof(float) * 3 * 3));
  CudaSafeCall(cudaMemcpy(d_Rq, h_Rq, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice));
}

void PointCloudTransformer::ConvertLLA2NEmU_GPU() {
  LLA2NEmU_GPU(h_positions_buffer_ptr, d_positions_buffer_ptr, 
               EQUATORIAL_RADIUS, eccentricity, buffer_size, 
               POSITION_DIM, reference_cam);
  CudaCheckError();
}

void PointCloudTransformer::ConvertNEmU2CamCoord_GPU() {
  float a = 1.0f, b = 0.0f;
  CublasSafeCall(cublasSgemm_v2(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                3, buffer_size, 3, &a, d_Rq, 3,
                                d_positions_buffer_ptr, 3, &b,
                                d_positions_buffer_ptr, 3));
}

void PointCloudTransformer::ConvertCamCoord2Img_CPU(int resolution) {
  if (img_allocated) {
    free(h_front);
    free(h_rear);
    free(h_left);
    free(h_right);
  }
  int img_size = sizeof(float) * resolution * resolution;
  h_front = (float *)malloc(img_size);
  h_rear = (float *)malloc(img_size);
  h_left = (float *)malloc(img_size);
  h_right = (float *)malloc(img_size);
  img_allocated = true;

  memset(h_front, 0, img_size);
  memset(h_rear, 0, img_size);
  memset(h_left, 0, img_size);
  memset(h_right, 0, img_size);

  LoadResults();

  for (int point_idx = 0; point_idx < buffer_size; point_idx++) {
    int cam_x_idx = point_idx * 3;
    float x = h_positions_buffer_ptr[cam_x_idx];
    float y = h_positions_buffer_ptr[cam_x_idx + 1];
    float z = h_positions_buffer_ptr[cam_x_idx + 2];

    int projected_x, projected_y;

    if (z < 0 && z < -abs(x) && z < -abs(y)) { //LEFT
      projected_x = resolution - (int)(((float)(-y / z) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;
      projected_y = (int)(((float)(-x / z) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;

      int projected_idx = projected_y * resolution + projected_x;
      h_left[projected_idx] += h_intensities_buffer_ptr[point_idx];
    }
    else if (z > 0 && z > abs(x) && z > abs(y)) { //RIGHT
      projected_x = resolution - (int)(((float)(y / z) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;
      projected_y = (int)(((float)(x / z) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;

      int projected_idx = projected_y * resolution + projected_x;
      h_right[projected_idx] += h_intensities_buffer_ptr[point_idx];
    }
    else if (x > 0 && x > abs(z) && x > abs(y)) { //FRONT
      projected_x = (int)(((float)(z / x) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;
      projected_y = resolution - (int)(((float)(y / x) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;

      int projected_idx = projected_y * resolution + projected_x;
      h_front[projected_idx] += h_intensities_buffer_ptr[point_idx];
    }
    else if (x < 0 && x < -abs(z) && x < -abs(y)) { //REAR
      projected_x = (int)(((float)(-z / x) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;
      projected_y = resolution - (int)(((float)(-y / x) * (resolution - 1) / 2)
                          + ((float)(resolution + 1) / 2)) - 1;

      int projected_idx = projected_y * resolution + projected_x;
      h_rear[projected_idx] += h_intensities_buffer_ptr[point_idx];
    }
  }

  cv::Mat img_scaled_left = cv::Mat(resolution, resolution, CV_8UC1);
  cv::Mat img_scaled_right = cv::Mat(resolution, resolution, CV_8UC1);
  cv::Mat img_scaled_front = cv::Mat(resolution, resolution, CV_8UC1);
  cv::Mat img_scaled_rear = cv::Mat(resolution, resolution, CV_8UC1);

  for (int i = 0; i < resolution; i++) {
    for (int j = 0; j < resolution; j++) {
      img_scaled_left.at<uchar>(j, i) = h_left[i * resolution + j];
      img_scaled_right.at<uchar>(j, i) = h_right[i * resolution + j];
      img_scaled_front.at<uchar>(i, j) = h_front[i * resolution + j];
      img_scaled_rear.at<uchar>(i, j) = h_rear[i * resolution + j];
    }
  }

  //show_img(img_scaled_front);

  cv::imwrite("img_left_unnormalized_" + std::to_string(resolution) 
              + ".jpg", img_scaled_left);
  cv::imwrite("img_right_unnormalized_" + std::to_string(resolution)
              + ".jpg", img_scaled_right);
  cv::imwrite("img_front_unnormalized_" + std::to_string(resolution)
              + ".jpg", img_scaled_front);
  cv::imwrite("img_rear_unnormalized_" + std::to_string(resolution)
              + ".jpg", img_scaled_rear);
  //Mat src = img_scaled_front;
  //Mat dst, cdst;
  ///* Set Region of Interest */

  //int offset_x = 0;
  //int offset_y = 0;

  //cv::Rect roi;
  //roi.x = offset_x;
  //roi.y = offset_y;
  //roi.width = src.size().width - (offset_x);
  //roi.height = src.size().height - (offset_y);

  ///* Crop the original image to the defined ROI */

  //cv::Mat crop = src(roi);
  //Canny(crop, dst, 60, 200, 3);
  //cvtColor(dst, cdst, CV_GRAY2BGR);

  //Mat kernel = Mat::zeros(5, 5, CV_8U); // 5x5 zero array
  //kernel.diag() = 1;                  // set 2nd row to '1'

  //cv::dilate(cdst, cdst, kernel);
  //cv::erode(cdst, cdst, kernel);
  //cv::dilate(cdst, cdst, kernel);
  //cv::erode(cdst, cdst, kernel);


  //vector<Vec4i> lines;
  //HoughLinesP(dst, lines, 1, CV_PI / 180, 130, 50, 20);
  //for (size_t i = 0; i < lines.size(); i++) {
  //  Vec4i l = lines[i];

  //  // oran = float(l[1] / col_size );
  //  double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
  //  if (angle < 0) angle += 180;
  //  if (angle > 180 && angle < 360) angle -= 180;
  //  angle = 180 - angle;

  //  if (angle != 0 && angle != 90 && angle != 180)
  //    if (angle <= 65 && angle >= 40) {
  //      //if(1){
  //      cout << angle << endl;
  //      float slope = (l[3] - l[1]) / (l[2] - l[0]);
  //      int c = l[3] - slope*l[2];

  //      int x = -c / slope;

  //      cv::Point(x, 0);
  //      line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
  //    }
  //}

  ////imshow("source", src);
  ////show_img(cdst);
  //cv::imwrite("img_front_unnormalized_" + std::to_string(resolution)
  //            + "_edge_detect.jpg", cdst);

  //NormalizeMatrix_CPU(h_left, resolution * resolution);
  //NormalizeMatrix_CPU(h_right, resolution * resolution);
  //NormalizeMatrix_CPU(h_front, resolution * resolution);
  //NormalizeMatrix_CPU(h_rear, resolution * resolution);
  //
  //for (int i = 0; i < resolution; i++) {
  //  for (int j = 0; j < resolution; j++) {
  //    img_scaled_left.at<uchar>(i, j) = h_left[i * resolution + j];
  //    img_scaled_right.at<uchar>(i, j) = h_right[i * resolution + j];
  //    img_scaled_front.at<uchar>(i, j) = h_front[i * resolution + j];
  //    img_scaled_rear.at<uchar>(i, j) = h_rear[i * resolution + j];
  //  }
  //}
  //cv::imwrite("img_left_normalized_" + std::to_string(resolution)
  //            + ".jpg", img_scaled_left);
  //cv::imwrite("img_right_normalized_" + std::to_string(resolution)
  //            + ".jpg", img_scaled_right);
  //cv::imwrite("img_front_normalized_" + std::to_string(resolution)
  //            + ".jpg", img_scaled_front);
  //cv::imwrite("img_rear_normalized_" + std::to_string(resolution)
  //            + ".jpg", img_scaled_rear);
}

//FAULTY!!!!
void PointCloudTransformer::ConvertCamCoord2Img_GPU(int resolution) {
  if (img_allocated) {
    CudaSafeCall(cudaFree(d_front));
    CudaSafeCall(cudaFree(d_rear));
    CudaSafeCall(cudaFree(d_left));
    CudaSafeCall(cudaFree(d_right));
    free(h_front);
    free(h_rear);
    free(h_left);
    free(h_right);
  }
  int img_size = sizeof(float) * resolution * resolution;
  CudaSafeCall(cudaMalloc((void **)&d_front, img_size));
  CudaSafeCall(cudaMalloc((void **)&d_rear, img_size));
  CudaSafeCall(cudaMalloc((void **)&d_left, img_size));
  CudaSafeCall(cudaMalloc((void **)&d_right, img_size));
  h_front = (float *)malloc(img_size);
  h_rear = (float *)malloc(img_size);
  h_left = (float *)malloc(img_size);
  h_right = (float *)malloc(img_size);
  img_allocated = true;

  FloatGPUMemset(d_front, resolution * resolution, 0.0F);
  CudaCheckError();
  FloatGPUMemset(d_rear, resolution * resolution, 0.0F);
  CudaCheckError();
  FloatGPUMemset(d_left, resolution * resolution, 0.0F);
  CudaCheckError();
  FloatGPUMemset(d_right, resolution * resolution, 0.0F);
  CudaCheckError();

  CamCoord2Img_GPU(d_positions_buffer_ptr, h_intensities_buffer_ptr,
                   d_front, d_rear, d_left, d_right, resolution,
                   buffer_size);
  CudaSafeCall(cudaMemcpy(h_front, d_front, img_size, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(h_rear, d_rear, img_size, cudaMemcpyDeviceToHost));
  //NormalizeMatrix_CPU(h_front, resolution * resolution);
  
  cv::Mat img_scaled = cv::Mat(resolution, resolution, CV_8UC1);

  for (int i = 0; i < resolution; i++) {
    for (int j = 0; j < resolution; j++) {
      img_scaled.at<uchar>(i, j) = h_front[i * resolution + j];
    }
  }
  cv::imwrite("img.jpg", img_scaled);
  show_img(img_scaled);
  CudaCheckError();
}

void PointCloudTransformer::NormalizeMatrix_CPU(float *h_mat,
                                                int total_size,
                                                int scale) {
  float max_val = 0;
  for (int i = 0; i < total_size; i++) {
    if (h_mat[i] > max_val)
      max_val = h_mat[i];
  }
  for (int i = 0; i < total_size; i++) {
    float coeff = (float) (scale / max_val);
    h_mat[i] *= coeff;
  }
}

void PointCloudTransformer::LoadResults() {
  CudaSafeCall(cudaMemcpy(h_positions_buffer_ptr, d_positions_buffer_ptr,
                          sizeof(float) * POSITION_DIM * buffer_size,
                          cudaMemcpyDeviceToHost));
}

std::vector<float> PointCloudTransformer::split(const std::string &s,
                                                char delim) {
  std::vector<float> elems;
  split_(s, delim, elems);
  return elems;
}

bool PointCloudTransformer::ReadNextRow() {
  std::vector<float> point_vect;
  if (pointcloud_fstream.is_open()) {
    if (std::getline(pointcloud_fstream, read_line)) {
      if (local_row_idx >= buffer_size) {
        local_row_idx = 0;
      }
      point_vect = split(read_line, ' ');
      for (int i = 0; i < point_vect.size(); i++) {
        if (i < POSITION_DIM) {
          h_positions_buffer_ptr[local_row_idx * POSITION_DIM + i]
            = point_vect[i];
          //std::cout << h_positions_buffer_ptr[local_row_idx * POSITION_DIM + i] << ", ";
        }
        else {
          h_intensities_buffer_ptr[local_row_idx] = point_vect[i];
          //std::cout << "---->" << intensities_buffer_ptr[local_row_idx];
        }
      }
      //std::cout << endl;
      local_row_idx++;
      global_row_idx++;
      return true;
    }
    else {
      pointcloud_fstream.close();
      return false;
    }
  }
  return false;
}

void PointCloudTransformer::init_Rq(float *h_Rq_ptr) {
  h_Rq_ptr[0] = std::powf(reference_cam.Qs, 2) + std::powf(reference_cam.Qx, 2)
    - std::powf(reference_cam.Qy, 2) - std::powf(reference_cam.Qz, 2);
  h_Rq_ptr[1] = 2 * (reference_cam.Qx * reference_cam.Qy 
                 - reference_cam.Qs * reference_cam.Qz);
  h_Rq_ptr[2] = 2 * (reference_cam.Qx * reference_cam.Qz 
                 + reference_cam.Qs * reference_cam.Qy);
  h_Rq_ptr[3] = 2 * (reference_cam.Qx * reference_cam.Qy 
                 + reference_cam.Qs * reference_cam.Qz);
  h_Rq_ptr[4] = std::powf(reference_cam.Qs, 2) - std::powf(reference_cam.Qx, 2)
    + std::powf(reference_cam.Qy, 2) - std::powf(reference_cam.Qz, 2);
  h_Rq_ptr[5] = 2 * (reference_cam.Qy * reference_cam.Qz 
                 - reference_cam.Qs * reference_cam.Qx);
  h_Rq_ptr[6] = 2 * (reference_cam.Qx * reference_cam.Qz 
                 - reference_cam.Qs * reference_cam.Qy);
  h_Rq_ptr[7] = 2 * (reference_cam.Qy * reference_cam.Qz 
                 + reference_cam.Qs * reference_cam.Qx);
  h_Rq_ptr[8] = std::powf(reference_cam.Qs, 2) - std::powf(reference_cam.Qx, 2)
    - std::powf(reference_cam.Qy, 2) + std::powf(reference_cam.Qz, 2);
}

void PointCloudTransformer::split_(const std::string &s, char delim,
                                   std::vector<float> &elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(stod(item));
  }
}

PointCloudTransformer::~PointCloudTransformer() { }