#include <iostream>
#include <malloc.h>
#include <opencv2/opencv.hpp>
#include <vector>
//#include <algorithm>
#include <math.h>

using namespace cv;
using namespace std;
#define Pi 3.1415926

void AddNoise(Mat src, Mat &dst, const int sigma, const int type)
{   
	Mat noise(src.size(), src.type());
    
    switch (type)
    {
    case 0:
        {
            randn(noise, Scalar::all(0), Scalar::all(sigma));
        }
        break;
    case 1:
        {
            Mat noise_tmp(src.size(), src.type());
            randn(noise_tmp, Scalar::all(0), Scalar::all(sigma));
            randn(noise, Scalar::all(0), Scalar::all(sigma));
            noise = noise.mul(noise) + noise_tmp.mul(noise_tmp);
            sqrt(noise, noise);
        }
        break;
    case 2:
        {
            Mat noise_tmp(src.size(), src.type());
            randn(noise_tmp, Scalar::all(0), Scalar::all(sigma));
            randn(noise, Scalar::all(0), Scalar::all(sigma));
            noise = noise.mul(noise) + noise_tmp.mul(noise_tmp);
        }
        break;
    case 3:
        {
            Mat noise_tmp(src.size(), src.type(), Scalar::all(sigma));
            noise_tmp.copyTo(noise);
        }
        break;
    case 4:
        {
            int A = sigma;
            double t = 0, dt = 0.05*Pi;
            int row = noise.rows, col = noise.cols, channel = noise.channels();
            float* data_noise = (float*)noise.data;
            for (int y = 0; y < row; y++) {
                for (int x = 0; x < col; x++) {
                    for (int z = 0; z < channel; z++) {
                        *data_noise++ += A * sin(t);
                    }
                    t += dt;
                }
            }
        }
        break;
    default:
        noise = Mat::zeros(src.size(), src.type());
        break;
    }
    dst = src + noise;
    //float *data_src = (float*)(src.data);
    //float *data_noise = (float*)noise.data;
    //float *data_dst = (float*)dst.data;

	//for (int j = 0; j < row; j++){
    //    for (int i = 0; i < col; i++){
    //        *data_dst++ = *data_src++ + *data_noise++;
    //    }
    //}
}

void addImpulseNoise(Mat& im, int n)
{
    for (int k = 0; k < n; ++k)
    {
        int i = rand() % im.rows;
        int j = rand() % im.cols;
        if(im.channels() == 1)
            im.at<uchar>(i, j) = 255;
        else
        {
            im.at<Vec3b>(i, j)[0] = 255;
            im.at<Vec3b>(i, j)[1] = 255;
            im.at<Vec3b>(i, j)[2] = 255;
        }
    }
    for (int k = 0; k < n; ++k)
    {
        int i = rand() % im.rows;
        int j = rand() % im.cols;
        if (im.channels() == 1)
            im.at<uchar>(i, j) = 0;
        else
        {
            im.at<Vec3b>(i, j)[0] = 0;
            im.at<Vec3b>(i, j)[1] = 0;
            im.at<Vec3b>(i, j)[2] = 0;
        }
    }
}

uchar adaptiveProcess(const Mat& im,const int row,const int col, int kernelSize,const int maxSize)
{
    vector<uchar> pixels;
    for (int a = -(kernelSize-1) / 2; a <= (kernelSize-1) / 2; a++)
        for (int b = -(kernelSize-1) / 2; b <= (kernelSize-1) / 2; b++)
        {
            pixels.push_back(im.at<uchar>(row + a, col + b));
        }
    sort(pixels.begin(), pixels.end());

    uchar min = pixels[0];
    uchar max = pixels[kernelSize * kernelSize - 1];
    uchar med = pixels[(kernelSize*kernelSize + 1) / 2];
    uchar zxy = im.at<uchar>(row, col);
    if (med > min && med < max)
    {
        // to level 2
        if (zxy > min && zxy < max)
            return zxy;
        else
            return med;
    }
    else
    {
        kernelSize += 2;
        if (kernelSize <= maxSize)
            return adaptiveProcess(im, row, col, kernelSize, maxSize); // 增大窗口尺寸，继续A过程。
        else
            return med;
    } 
}

void adaptiveMediaFilter(const Mat& src, Mat &dst, const int maxSize, const int minSize)
{
    for (int i = (maxSize-1)/2; i < dst.rows - (maxSize - 1)/2; ++i)
    {
        for (int j = (maxSize - 1) / 2; j < dst.cols - (maxSize - 1) / 2; ++j)
        {
            dst.at<uchar>(i, j) = adaptiveProcess(dst, i, j, minSize, maxSize);
        }
    }   
}

int NLM_withsigma_gray(Mat& src_i, Mat& dst, double sigma, double h = 10, int type = 1, int r = 3, int R = 10) {
    int d = 2 * r + 1, D = 2 * R + 1;
    double sigma2 = sigma * sigma, dh2 = 1 / (h * h);
    Mat src;
    copyMakeBorder(src_i, src, R + r, R + r, R + r, R + r, BORDER_REFLECT);
    Mat kernel;
    switch (type) {
    case 0:
        kernel = kernel.ones(2 * r + 1, 2 * r + 1, CV_64F) / ((2 * r + 1) * (2 * r + 1));
        break;
    case 1:
        kernel = getGaussianKernel(d, 1) * getGaussianKernel(d, 1).t();
        break;
    default:
        return 1;
        break;
    }

    uchar* data_src = src.data;
    uchar* data_kernel = kernel.data, * data_dst = dst.data;

    int row_dst = src_i.rows, col_dst = src_i.cols, row_src = src.rows, col_src = src.cols;
    int x_dst, y_dst, x_src, y_src;
    int x_search, y_search, x_template, y_template;
    double w, w_center, w_total;
    double d_eu, cal;
    double average;
    float rate;
    for (y_dst = 0; y_dst < row_dst; y_dst++) {
        if (y_dst % 100 == 0) { rate = (float)y_dst / (float)row_dst * 100; cout << rate << "%" << endl; }
        for (x_dst = 0; x_dst < col_dst; x_dst++) {           
            x_template = x_dst + R;
            y_template = y_dst + R;
            //初始化待处理
            w = 0;
            w_center = 0;
            w_total = 0;
            average = 0;
            //
            for (y_search = 0; y_search < D; y_search++) {
                for (x_search = 0; x_search < D; x_search++) {
                    //计算欧式距离
                    d_eu = 0; cal = 0;//cal需要初始化么
                    for (int j = 0; j < d; j++) {
                        for (int i = 0; i < d; i++) {
                            if ((x_search + i == R + r) && (y_search + j == R + r)) continue;
                            
                            cal = (int)(*(data_src + (y_template + j) * src.step[0] + (x_template + i) * src.step[1])) - (int)(*(data_src + (y_dst + y_search + j) * src.step[0] + (x_dst + x_search + i) * src.step[1]));
                            cal *= cal;
                            
                            d_eu += cal * (*(double*)(data_kernel + j * kernel.step[0] + i * kernel.step[1]));
                        }
                    }
                    
                    d_eu = (d_eu - 2 * sigma2 > 0) ? d_eu : 0;
                    w = exp(-d_eu * dh2);
                    w_center = w > w_center ? w : w_center;
                    w_total += w;
                    
                    average += w * (int)(*(data_src + (y_dst + y_search + r) * src.step[0] + (x_dst + x_search + r) * src.step[1]));
                }
            }
            w_total += w_center;
            
            average += w_center * (int)(*(data_src + (y_template + r) * src.step[0] + (x_template + r) * src.step[1]));
            if (w_total > 0)
                *(data_dst + y_dst * dst.step[0] + x_dst * dst.step[1]) = (uchar)(average / w_total);
            else
                *(data_dst + y_dst * dst.step[0] + x_dst * dst.step[1]) = *(data_src + (y_template + r) * src.step[0] + (x_template + r) * src.step[1]);
        }
    }
    return 0;
}

int NLM_withsigma(Mat& src_i, Mat& dst, double sigma, double h = 10, int type = 1, int r = 3, int R = 10) {
    int d = 2 * r + 1, D = 2 * R + 1;
    double sigma2 = sigma * sigma, dh2 = 1 / (h * h);
    int channel = src_i.channels();
    Mat src;
    copyMakeBorder(src_i, src, R + r, R + r, R + r, R + r, BORDER_REFLECT);
    Mat kernel;
    switch (type) {
    case 0:
        kernel = kernel.ones(2 * r + 1, 2 * r + 1, CV_64F) / ((2 * r + 1) * (2 * r + 1));
        break;
    case 1:
        kernel = getGaussianKernel(d, 1) * getGaussianKernel(d, 1).t();
        break;
    default:
        return 1;
        break;
    }

    uchar* data_src = src.data;
    uchar* data_kernel = kernel.data, * data_dst = dst.data;

    int row_dst = src_i.rows, col_dst = src_i.cols, row_src = src.rows, col_src = src.cols;
    int x_dst, y_dst, x_src, y_src;
    int x_search, y_search, x_template, y_template;
    double w, w_center, w_total;
    double d_eu, cal;
    double* average = (double*)malloc(sizeof(double) * channel);
    float rate;
    for (y_dst = 0; y_dst < row_dst; y_dst++) {
        if (y_dst % 100 == 0) { rate = (float)y_dst / (float)row_dst * 100; cout << rate << "%" << endl; }
        for (x_dst = 0; x_dst < col_dst; x_dst++) {
            x_template = x_dst + R;
            y_template = y_dst + R;

            //初始化
            w = 0;
            w_center = 0;
            w_total = 0;
      
            for (int z = 0; z < channel; z++) { average[z] = 0; }

            for (y_search = 0; y_search < D; y_search++) {
                for (x_search = 0; x_search < D; x_search++) {
                    //计算欧式距离
                    d_eu = 0; cal = 0;
                    for (int j = 0; j < d; j++) {
                        for (int i = 0; i < d; i++) {
                            if ((x_search + i == R + r) && (y_search + j == R + r)) continue;
                            for (int z = 0; z < channel; z++) {
                                cal = (int)(*(data_src + (y_template + j) * src.step[0] + (x_template + i) * src.step[1] + z)) - (int)(*(data_src + (y_dst + y_search + j) * src.step[0] + (x_dst + x_search + i) * src.step[1] + z));
                                cal *= cal;
                                d_eu += cal * (*(double*)(data_kernel + j * kernel.step[0] + i * kernel.step[1]));
                            }
                        }
                    }
                    d_eu /= channel;
                    d_eu = (d_eu - 2 * sigma2 > 0) ? d_eu : 0;
                    w = exp(-d_eu * dh2);
                    w_center = w > w_center ? w : w_center;
                    w_total += w;

                    for (int z = 0; z < channel; z++) {
                        average[z] += w * (int)(*(data_src + (y_dst + y_search + r) * src.step[0] + (x_dst + x_search + r) * src.step[1] + z));
                    }
                }
            }
            w_total += w_center;
            for (int z = 0; z < channel; z++) {
                average[z] += w_center * (int)(*(data_src + (y_template + r) * src.step[0] + (x_template + r) * src.step[1] + z));
            }

            if (w_total > 0)
                for (int z = 0; z < channel; z++) {
                    *(data_dst + y_dst * dst.step[0] + x_dst * dst.step[1] + z) = (uchar)(average[z] / w_total);
                }
            else
                for (int z = 0; z < channel; z++) {
                    *(data_dst + y_dst * dst.step[0] + x_dst * dst.step[1] + z) = *(data_src + (y_template + r) * src.step[0] + (x_template + r) * src.step[1] + z);
                }
        }
    }
    return 0;
}

int Local_Mean_filter(Mat& src_i, Mat& dst, int type = 1, int h = 1, int r = 1) {
    int d = 2 * r + 1;
    Mat src;
    copyMakeBorder(src_i, src, r, r, r, r, BORDER_REFLECT);
    
    Mat kernel;
    switch (type) {
    case 0:
        kernel = kernel.ones(d, d, CV_64F) / (d * d);
        break;
    case 1:
        kernel = getGaussianKernel(d, 1) * getGaussianKernel(d, 1).t();
        break;
    default:
        return 1;
        break;
    }
    
    int channel = src_i.channels();
    int row_dst = src_i.rows, col_dst = src_i.cols;//, row_src = src.rows, col_src = src.cols;
    uchar *data_src = src.data, *data_dst = dst.data, *data_kernel = kernel.data;
    double average;
    for (int y = 0; y < row_dst; y++) {
        for (int x = 0; x < col_dst; x++) {
            for (int z = 0; z < channel; z++) {
                average = 0;
                for (int j = 0; j < d; j++) {
                    for (int i = 0; i < d; i++) {
                        average += (*(double*)(data_kernel + j * kernel.step[0] + i * kernel.step[1])) * (int)(*(data_src + (y + j) * src.step[0] + (x + i) * src.step[1] + z));
                    }
                }
                *(data_dst + y * dst.step[0] + x * dst.step[1] + z) = (uchar)average;
            }
        }
    }

    if (--h > 0) {
        Mat src_o = dst;
        Local_Mean_filter(src_o, dst, h, type, r);
    }
    else
        return 0;
}

int Derivation(Mat& src_i, Mat& dst, int dx, int dy) {
    int channel = src_i.channels();
    //cout << channel<<endl;
    Mat src;
    copyMakeBorder(src_i, src, 0, dy, 0, dx, BORDER_REFLECT);
    int row_dst = src_i.rows, col_dst = src_i.cols;
    uchar *data_dst = dst.data, *data_src = src.data;
    
    int total, tmp;
    for (int y = 0; y < row_dst; y++) {
        for (int x = 0; x < col_dst; x++) {
            total = 0;           
            tmp = 0;
            for (int z = 0; z < channel; z++) {
                tmp = (*(int*)(data_src + (y + dy) * src.step[0] + (x + dx) * src.step[1] + z)) - (*(int*)(data_src + y * src.step[0] + x * src.step[1] + z));
                total += tmp >= 0 ? tmp : -tmp;
            }     
            *(data_dst + y * dst.step[0] + x * dst.step[1])= total / channel;
        }
    }

    return 0;
}

double Standard_Deviation_Estimator(Mat& src, int type = 0) {
    int channel = src.channels();

    double variance=-1;
    double standard_deviation=-1;
    int row = src.rows, col = src.cols;
    //uchar *dat = src.data;
    int kernel[9] = { 1, -2, 1, -2, 4, -2, 1, -2, 1 };

    int cal = 0;
    long long int total = 0;
    if (type == 0) {
        for (int y = 0; y < row - 2; y++) {
            for (int x = 0; x < col - 2; x++) {
                cal = 0;
                for (int j = 0; j < 3; j++) {
                    for (int i = 0; i < 3; i++) {
                        for (int z = 0; z < channel; z++) {
                            cal += kernel[j * 3 + i] * (int)(*(src.data + (y + j) * src.step[0] + (x + i) * src.step[1] + z));
                        }
                    }
                }
                total += cal > 0 ? cal : -cal;
            }
        }
        //total /= sqrt(channel);
        standard_deviation = total * (sqrt(Pi / 2)) / 6 / (row - 2) / (col - 2) / sqrt(channel);
    }

    else {
        for (int y = 0; y < row - 2; y++) {
            for (int x = 0; x < col - 2; x++) {
                cal = 0;
                for (int j = 0; j < 3; j++) {
                    for (int i = 0; i < 3; i++) {
                        for (int z = 0; z < channel; z++) {
                            cal += kernel[j * 3 + i] * (int)(*(src.data + (y + j) * src.step[0] + (x + i) * src.step[1] + z));
                        }
                    }
                }
                total += cal * cal;
            }
        }
        variance = total / 36 / (row - 2) / (col - 2) / channel;
        //variance /= channel;
        //standard_deviation = sqrt(variance);
    }

    if (type == 0)
        return standard_deviation;
    else
        return variance;
}

int Adaptive_Mean_Filter(Mat &src_i, Mat &dst, double sigma_n = 0, int r = 1, int type = 0){
    int channel = src_i.channels();
    int d = 2*r+1;
    int N = d*d;
    double sigma_n2 = sigma_n * sigma_n;

    Mat src;
    copyMakeBorder(src_i, src, r, r, r, r, BORDER_REFLECT);
    int row = src_i.rows, col = src_i.cols;

    if (type == 0){//基于局部方差
        int total, cal;
        double average, sigma_area2;
        //double lower_base = 0.9, higher_base = 1.1;
        double ratio = 0;
        for (int y = 0; y < row; y++){
            for (int x = 0; x < col; x++){
                for (int z = 0; z < channel; z++){
                /*计算均值*/
                total = 0;
                average = 0;
                for (int j = 0; j < d; j++){
                    for (int i = 0; i < d; i++){
                        total += (int)(*(src.data + (y+j)*src.step[0] + (x+i)*src.step[1] + z));
                    }
                }
                average = total/(d*d);
                /*计算方差*/
                total = 0;
                cal = 0;
                sigma_area2 = 0;
                for (int j = 0; j < d; j++){
                    for (int i = 0; i < d; i++){
                        cal = (int)(*(src.data + (y+j)*src.step[0] + (x+i)*src.step[1] + z)) - average;
                        total += cal*cal;
                    }
                }
                sigma_area2 = total/(d*d);

                ratio = (sigma_n2/sigma_area2) > 1 ? 1 : (sigma_n2/sigma_area2);
                *(dst.data + y*dst.step[0] + x*dst.step[1] + z) = (uchar)((int)(*(src.data + (y+r)*src.step[0] + (x+r)*src.step[1] + z))*(1-ratio) + ratio*average);
                }
            }
        }
    }

    else {//基于中值的加权均值滤波
        double weight, weight_total;
        double total;
        int med;
        double cal;
        double T;
        int *area = (int*)malloc(sizeof(int)*N);
        for (int y = 0; y < row; y++){
            for (int x = 0; x < col; x++){
                for (int z = 0; z < channel; z++){
                /*计算中值*/
                for (int j = 0; j < d; j++){
                    for (int i = 0; i < d; i++){
                        *(area + j*d + i) = (int)(*(src.data + (y+j)*src.step[0] + (x+i)*src.step[1] + z));
                    }
                }

                med = area[0];
                cal = area[0];
                for (int i = 0; i < N; i++){
                    for (int j = 1; j < N; j++){
                        if (area[i] > area[j]){
                            cal = area[i];
                            area[i] = area[j];
                            area[j] = cal;
                        }
                    }
                }
                med = area[(int)(N/2)];

                /*计算窗内像素与中值差平方的均值*/
                total = 0;
                cal = 0;
                T = 0;
                for (int i = 0; i < N; i++){
                    cal = area[i] - med;
                    total += cal*cal;
                }
                T = total/N;

                /*计算权值*/
                total = 0;
                cal = 0;
                weight_total = 0;
                weight = 0;
                for (int i = 0; i < N; i++){
                    cal = (area[i] - med) * (area[i] - med);
                    cal = T>cal ? T : cal;
                    weight = 1/(1+cal);
                    //weight = exp(-cal/10);
                    weight_total += weight;
                    total += weight * area[i];
                }
             
                *(dst.data + y*dst.step[0] + x*dst.step[1] + z) = (uchar)(total/weight_total);
                }
            }
        }
    }
    return 0;
}

double MSE(Mat &img_a, Mat &img_b){
    int channel = img_a.channels();
    int row = img_a.rows, col = img_a.cols;
    uchar *data_a = img_a.data, *data_b = img_b.data;

    if ((row!=img_b.rows)||(col!=img_b.cols)||(channel!=img_b.channels()))
        return -1;
    else{
        double mse = -1;
        int num = row * col * channel;
        int cal = 0;
        long long int total = 0;

        
        for (int i = 0; i < num; i++) {
            cal = (int)(*data_a++) - (int)(*data_b++);
            total += cal * cal;
        }

        mse = total / num;

        return mse;
        /*
        for (int y = 0; y < row; y++) {
            for (int x = 0; x < col; x++) {
                for (int z = 0; z < channel; z++) {
                    cal = (int)(*(data_a + y * img_a.step[0] + x * img_a.step[1] + z)) - (int)(*(data_b + y * img_b.step[0] + x * img_b.step[1] + z));
                    total += cal * cal;
                }
            }
        }
        MSE = total / row / col / channel;
        return MSE;
        */
    }
}

double PSNR(Mat &img_a, Mat &img_b){
    if ((img_a.rows != img_b.rows) || (img_a.cols != img_b.cols) || (img_a.channels() != img_b.channels()))
        return -1;
    else {
        double psnr = -1;
        double mse = MSE(img_a, img_b);
        psnr = 10*log10(255*255/mse);
        return psnr;
    }
}

double SSIM(Mat &img_a, Mat &img_b){
    if ((img_a.rows != img_b.rows) || (img_a.cols != img_b.cols) || (img_a.channels() != img_b.channels()))
        return -1;
    else{
        int row = img_a.rows, col = img_a.cols, channel = img_a.channels();
        uchar *data_a = img_a.data, *data_b = img_b.data;
        int step[2] = {img_a.step[0], img_a.step[1]};

        double K1 = 0.01, K2 = 0.03;
        int L = 255;
        double C1 = (K1*L)*(K1*L), C2 = (K2*L)*(K2*L);
        double ssim = 0;

        double u_a = 0, u_b = 0;
        double sigma2_a = 0, sigma2_b = 0, sigma_ab = 0;
        //double sigma_a = 0, sigma_b = 0;

        double cal_a = 0, cal_b = 0;
        double total_a = 0, total_b = 0, total_ab = 0;
        for (int z = 0; z < channel; z++){

            u_a = u_b = 0;
            for (int y = 0; y < row; y++){
                for (int x = 0; x < col; x++){
                    total_a += (int)(*(data_a + y*step[0] + x*step[1] + z));
                    total_b += (int)(*(data_b + y*step[0] + x*step[1] + z));
                }
            }
            u_a = total_a / (row*col);
            u_b = total_b / (row*col);

            sigma2_a = sigma2_b = sigma_ab = 0;
            //sigma_a = sigma_b = 0
            total_a = total_b = total_ab = 0;
            cal_a = cal_b = 0;
            for (int y = 0; y < row; y++){
                for (int x = 0; x < col; x++){
                    cal_a = (int)(*(data_a + y*step[0] + x*step[1] + z)) - u_a;
                    cal_b = (int)(*(data_b + y*step[0] + x*step[1] + z)) - u_b;
                    total_a += cal_a * cal_a;
                    total_b += cal_b * cal_b;
                    total_ab += cal_a * cal_b;
                }
            }
            sigma2_a = total_a / (row*col);
            sigma2_b = total_b / (row*col);
            sigma_ab = total_ab / (row*col);

            //sigma_a = sqrt(sigma2_a);
            //sigma_b = sqrt(sigma2_b);

            ssim += (2*u_a*u_b + C1)*(2*sigma_ab + C2)/(u_a*u_a + u_b*u_b + C1)/(sigma2_a + sigma2_b + C2);
        }
        ssim /= channel;

        return ssim;
    }
}

int log2(const int N)
{
	int k = 1;
	int n = 0;
	while (k < N)
	{
		k *= 2;
		n++;
	}
	return n;
}

void wavedec(float *input, int length)
{
	int N = log2(length);
	if (N == 0 )return;
	float *tmp = new float[length];
	for (int i = 0; i < length; i++){
		tmp[i] = input[i];
	}
	for (int k = 0; k <length/2; k++)
	{
		input[k] = (tmp[2*k] + tmp[2*k + 1]) / sqrt(2);
		input[k + length/2] = (tmp[2*k] - tmp[2*k + 1])/sqrt(2);
	}
	delete [] tmp;
	wavedec(input, length / 2);
	return;
}

void waverec(float* input, int length, int N)
{
	if (log2(length) > N) return;
	float *tmp = new float[length];
	for (int i = 0; i < length; i++){
		tmp[i] = input[i];
	}
	for (int k = 0; k < length / 2; k++)
	{
		input[2 * k] = (tmp[k]+tmp[k+length/2])/sqrt(2);
		input[2 * k+1] = (tmp[k] - tmp[k + length / 2]) / sqrt(2);
	}
	delete tmp;
	waverec(input, length * 2, N);
}

void gen_wienFilter(vector<Mat>&wien, int sigma)
{
	Mat tmp;
	Mat Sigma(wien[0].size(), CV_32FC1, Scalar::all(sigma*sigma));
	for (int k = 0; k < wien.size(); k++)
	{
		tmp = wien[k].mul(wien[k]) + Sigma;
		wien[k] = wien[k].mul(wien[k]) / (tmp.clone());
	}
}

void wienFiltering(vector<Mat>&input, const vector<Mat>wien, int patchSize)
{
	for (int k = 0; k < input.size(); k++)
	{
		input[k] = input[k].mul(wien[k]);
	}
}

Mat Kaiser(int beta,int length)
{
	if ((beta == 2)&&(length==8))
	{
		Mat window(length, length, CV_32FC1);
		Mat kai1(length, 1, CV_32FC1);
		Mat kai1_T(1, length, CV_32FC1);
		kai1.at<float>(0, 0) = 0.4387;
		kai1.at<float>(1, 0) = 0.6813;
		kai1.at<float>(2, 0) = 0.8768;
		kai1.at<float>(3, 0) = 0.9858;
		for (int i = 0; i < 4; i++)
		{
			kai1.at<float>(7 - i, 0) = kai1.at<float>(i,0);
			kai1_T.at<float>(0, i) = kai1.at<float>(i, 0);
			kai1_T.at<float>(0, 7 - i) = kai1.at<float>(i, 0);
		}
		window = kai1*kai1_T;
		return window;
	}
}

float cal_distance(Mat a, Mat b)
{
	int sy = a.rows;
	int sx = a.cols;
	float sum = 0;
	for (int i = 0; i < sy; i++)
	{
		const float* M1 = a.ptr<float>(i);
		const float* M2 = b.ptr<float>(i);
		for (int j = 0; j < sx; j++)
		{
			sum += (M1[j] - M2[j])*(M1[j] - M2[j]);
		}
	}
	return sum / (sy*sx);
}

void getPatches(const Mat &img, const int width, const int height, const int channels,
	const int patchSize, const int step, vector<Mat>&block, vector<int>&row_idx, vector<int>&col_idx)
{
	Mat tmp(patchSize, patchSize, CV_32FC1);
	for (int i = 0; i <= height - patchSize; i += step)
	{
		row_idx.push_back(i);
	}
	if ((height - patchSize) % step != 0)
	{
		row_idx.push_back(height - patchSize);
	}
	for (int j = 0; j <= width - patchSize; j += step)
	{
		col_idx.push_back(j);
	}
	if ((width - patchSize) % step != 0)
	{
		col_idx.push_back(width - patchSize);
	}
	for (int i = 0; i < row_idx.size(); i++)
	{
		for (int j = 0; j < col_idx.size(); j++)
		{
			tmp = img(Rect(col_idx[j],row_idx[i], patchSize, patchSize));
			block.push_back(tmp);
		}
	}
}

void getPatches(const Mat &img, const int patchSize, vector<Mat>&block, vector<int>&row_idx, vector<int>&col_idx)
{
	Mat tmp(patchSize, patchSize, CV_32FC1);
	for (int i = 0; i < row_idx.size(); i++)
	{
		for (int j = 0; j < col_idx.size(); j++)
		{
			tmp = img(Rect(col_idx[j],row_idx[i], patchSize, patchSize));
			block.push_back(tmp);
		}
	}
}

void getSimilarPatch(const vector<Mat> block, vector<Mat>&sim_patch, vector<int>&sim_num,
	int i, int j, int bn_r, int bn_c, int area, int maxNum, int tao)
{
	int row_min = max(0, i - (area - 1) / 2);
	int row_max = min(bn_r-1, i + (area - 1) / 2);
	int row_length = row_max - row_min + 1;

	int col_min = max(0, j - (area - 1) / 2);
	int col_max = min(bn_c-1, j + (area - 1) / 2);
	int col_length = col_max - col_min + 1;

	const Mat relevence = block[i*bn_c + j];
	Mat tmp;
	 
	float* distance = new float[row_length*col_length];//计算距离
	int* idx = new int[row_length*col_length];//保存下标便于后续聚类计算
	if (!distance){
		cout << "allocation failure\n";
		system("pause");
	}
	for (int p = 0; p <row_length; p++)
	{
		for (int q = 0; q < col_length; q++)
		{
			tmp = block[(p + row_min)*bn_c + (q + col_min)];
			distance[p*col_length + q] = cal_distance(relevence, tmp);
			//cout << distance[p*col_length + q] << endl;
			idx[p*col_length + q] = p*col_length + q;
		}
	}
	float value; int l;
	//直接排序算法，有待改进！
	for (int k = 1; k < row_length*col_length; k++)
	{
		value = distance[k];
		for (l = k - 1; value < distance[l]&&l >= 0; --l)
		{
			distance[l + 1] = distance[l];
			idx[l + 1] = idx[l];
		}
		distance[l + 1] = value;
		idx[l + 1] = k;
	}
	int selectedNum = maxNum;
	while (row_length*col_length<selectedNum)
	{
		selectedNum /= 2;//确保相似块的个数为2的幂
	}
	while (distance[selectedNum - 1] > tao)
	{
		selectedNum /= 2;
	}
	int Row, Col;
	for (int k = 0; k < selectedNum; k++)
	{
		Row = row_min + idx[k] / col_length;
		Col = col_min + idx[k] % col_length;
		tmp = block[Row*bn_c + Col].clone();
		sim_patch.push_back(tmp);
		sim_num.push_back(Row*bn_c + Col);
	}
}

void getSimilarPatchNum(const vector<Mat> block, vector<int>&sim_num, int selectedNum,
	int i, int j, int bn_r, int bn_c, int area, int maxNum, int tao)
{
	int row_min = max(0, i - (area - 1) / 2);
	int row_max = min(bn_r-1, i + (area - 1) / 2);
	int row_length = row_max - row_min + 1;

	int col_min = max(0, j - (area - 1) / 2);
	int col_max = min(bn_c-1, j + (area - 1) / 2);
	int col_length = col_max - col_min + 1;

	const Mat relevence = block[i*bn_c + j];
	Mat tmp;
	 
	float* distance = new float[row_length*col_length];//计算距离
	int* idx = new int[row_length*col_length];//保存下标便于后续聚类计算
	if (!distance){
		cout << "allocation failure\n";
		system("pause");
	}
	for (int p = 0; p <row_length; p++)
	{
		for (int q = 0; q < col_length; q++)
		{
			tmp = block[(p + row_min)*bn_c + (q + col_min)];
			distance[p*col_length + q] = cal_distance(relevence, tmp);
			//cout << distance[p*col_length + q] << endl;
			idx[p*col_length + q] = p*col_length + q;
		}
	}
	float value; int l;
	//直接排序算法，有待改进！
	for (int k = 1; k < row_length*col_length; k++)
	{
		value = distance[k];
		for (l = k - 1; value < distance[l]&&l >= 0; --l)
		{
			distance[l + 1] = distance[l];
			idx[l + 1] = idx[l];
		}
		distance[l + 1] = value;
		idx[l + 1] = k;
	}
	selectedNum = maxNum;
	while (row_length*col_length<selectedNum)
	{
		selectedNum /= 2;//确保相似块的个数为2的幂
	}
	while (distance[selectedNum - 1] > tao)
	{
		selectedNum /= 2;
	}
	int Row, Col;
	for (int k = 0; k < selectedNum; k++)
	{
		Row = row_min + idx[k] / col_length;
		Col = col_min + idx[k] % col_length;
		sim_num.push_back(Row*bn_c + Col);
	}
}

void getSimilarPatch_withnum(const vector<Mat> block, vector<Mat>&sim_patch, vector<int>&sim_num, int selectedNum)
{
	Mat tmp;
	for (int k = 0; k < selectedNum; k++)
	{
		tmp = block[sim_num[k]].clone();
		sim_patch.push_back(tmp);
	}
}

void tran2d(vector<Mat> &input, const char* tran_mode,int patchsize)
{
	Mat tmp;
	if (tran_mode == "DCT")
	{
		int length = input.size();
		for (int i = 0; i < length; i++)
		{
			dct(input[i],tmp);
			input[i] = tmp.clone();
		}
	}
	else if (tran_mode == "BIOR1.5")
	{

	}
}

void tran1d(vector<Mat>&input, const char* tran_mode,int patchSize)
{
	//暂时只做了haar一维小波分解
	if (tran_mode == "HAAR")
	{
		int size = input.size();
		int layer = log2(size);
		float* data = new float[size];
		for (int i = 0; i < patchSize; i++)
		{
			for (int j = 0; j < patchSize; j++)
			{
				for (int k = 0; k < size; k++)
				{
					data[k] = input[k].at<float>(i, j);
					//cout << data[k] << endl;
				}
				wavedec(data, size);
				for (int k = 0; k < size; k++)
				{
					input[k].at<float>(i, j) = data[k];
				}
			}
		}
		delete[] data;
	}
}

void shrink(vector<Mat>&input, float threshold)
{
	for (int k = 0; k < input.size(); k++)
	{
		for (int i = 0; i < input[k].rows; i++)
			for (int j = 0; j <input[k].cols; j++)
			{
				if (fabs(input[k].at<float>(i, j)) < threshold)
				{
					input[k].at<float>(i, j) = 0;
				}
			}
	}
}

float calculate_weight_hd(const vector<Mat>input, int sigma)
{
	int num = 0;
	for (int k = 0; k < input.size(); k++)
		for (int i = 0; i < input[k].rows; i++)
			for (int j = 0; j < input[k].cols; j++)
			{
				if (input[k].at<float>(i, j) != 0)
				{
					num++;
				}
			}
	if (num == 0)
	{
		return 1;
	}
	else
		return 1.0 / (sigma*sigma*num);
}

float calculate_weight_wien(const vector<Mat>input, int sigma)
{
	float sum=0;
	for (int k = 0; k < input.size(); k++)
		for (int i = 0; i < input[k].rows; i++)
			for (int j = 0; j < input[k].cols; j++)
			{
				sum += (input[k].at<float>(i, j))*(input[k].at<float>(i, j));
			}
	return 1.0 / (sigma*sigma*sum);
}

void inv_tran_3d(vector<Mat>&input, const char* mode2d, const char *mode1d,int patchSize)
{
	if ((mode2d == "DCT") && (mode1d == "HAAR"))
	{
		Mat tmp;
		int size = input.size();
		int layer = log2(size);
		float* data = new float[size];
		for (int i = 0; i < patchSize; i++)
			for (int j = 0; j < patchSize; j++)
			{
				for (int k = 0; k < size; k++)
				{
					data[k] = input[k].at<float>(i, j);
				}
				waverec(data,2,layer);
				for (int k = 0; k < size; k++)
				{
					input[k].at<float>(i, j) = data[k];
				}
			}
		for (int k = 0; k < size; k++)
		{
			tmp = input[k].clone();
			dct(tmp, input[k], DCT_INVERSE);
		}
	}
}

void aggregation(Mat &numerator, Mat &denominator, vector<int>idx_r, vector<int>idx_c, 
	const vector<Mat> input, float weight, int patchSize,Mat window)
{
	Rect rect;
	for (int k = 0; k < input.size(); k++)
	{
		rect.x = idx_c[k];
		rect.y = idx_r[k];
		rect.height = patchSize;
		rect.width = patchSize;
		numerator(rect) = numerator(rect) + weight*(input[k].mul(window));
		denominator(rect) = denominator(rect) + weight*window;
	}
}


int BM3D_gray(Mat &src, Mat &img_basic, Mat &dst, double sigma){
    const int nHard = 39;//search area
	const int nWien = 39;
	const int kHard = 8;//patch size
	const int kWien = 8;
	const int NHard = 16;//max number
	const int NWien = 32;
	const int pHard = 3;//step
	const int pWien = 3;

	const int tao_hard = 2500;
	const int tao_wien = 400;

	const int beta = 2;
	const float lambda3d = 2.7;
	const float lambda2d = 0;

    int Height = src.rows;
	int Width = src.cols;
	int Channels = src.channels();

	//step 1 hard threshhold filtering
	/*if (Channels == 3)
	{
		cvtColor(src, src, CV_BGR2YUV);
	}*/
	vector<Mat> block_src;//store the patch
	vector<int>row_idx;//patch idx along the row direction
	vector<int>col_idx;
	getPatches(src, Width, Height, Channels, kHard, pHard,block_src,row_idx, col_idx);
	int bn_r = row_idx.size();
	int bn_c = col_idx.size();
	tran2d(block_src, "DCT",kHard);

	vector<int> sim_num;//index number for the selected similar patch in the block vector
	vector<int> sim_idx_row;//index number for the selected similar patch in the srcal Mat
	vector<int> sim_idx_col;

	vector<Mat>data;//store the data during transforming and shrinking

	Mat kaiser = Kaiser(beta, kHard);//2-D kaiser window 
	float weight_hd=1.0;//weights used for current relevent patch
	Mat denominator_hd(src.size(), CV_32FC1,Scalar::all(0));
	Mat numerator_hd(src.size(), CV_32FC1,Scalar::all(0));
	for (int i = 0; i < bn_r; i++)
	{
		for (int j = 0; j < bn_c; j++)
		{
			//for each pack in the block
			sim_num.clear();
			sim_idx_row.clear();
			sim_idx_col.clear();
			data.clear();

			getSimilarPatch(block_src, data, sim_num,
				i, j, bn_r,bn_c,int((nHard-kHard) / pHard)+1, NHard,tao_hard);//block matching

			for (int k = 0; k < sim_num.size(); k++)//calculate idx in the left-top corner
			{
				sim_idx_row.push_back(row_idx[sim_num[k] / bn_c]);
				sim_idx_col.push_back(col_idx[sim_num[k] % bn_c]);
			}

			tran1d(data ,"HAAR",kHard);//3-D transforming

			shrink(data,lambda3d*sigma);//shrink the cofficient

			weight_hd = calculate_weight_hd(data,sigma);

			inv_tran_3d(data, "DCT", "HAAR",kHard);//3-D inverse transforming
			
			aggregation(numerator_hd, denominator_hd, sim_idx_row, sim_idx_col, data,weight_hd,kHard,kaiser);//aggregation using weigths
		}
	}
	img_basic = numerator_hd / denominator_hd;

	//step 2 wiena filtering
	vector<Mat> block_basic;
	row_idx.clear();
	col_idx.clear();

	getPatches(img_basic, Width, Height, Channels, kHard, pHard, block_basic, row_idx, col_idx);
	bn_r = row_idx.size();
	bn_c = col_idx.size();

	vector<Mat> data_src;
	float weight_wien = 1.0;//weights used for current relevent patch
	Mat denominator_wien(src.size(), CV_32FC1, Scalar::all(0));
	Mat numerator_wien(src.size(), CV_32FC1, Scalar::all(0));
	for (int i = 0; i < bn_r; i++)
	{
		for (int j = 0; j < bn_c; j++)
		{
			//for each pack in the basic estimate
			sim_num.clear();
			sim_idx_row.clear();
			sim_idx_col.clear();
			data.clear();
			data_src.clear();

			getSimilarPatch(block_basic, data, sim_num,i, j, bn_r, bn_c, 
				int((nWien - kWien) / pWien) + 1, NWien, tao_wien);//block matching

			for (int k = 0; k < sim_num.size(); k++)//calculate idx in the left-top corner
			{
				sim_idx_row.push_back(row_idx[sim_num[k] / bn_c]);
				sim_idx_col.push_back(col_idx[sim_num[k] % bn_c]);
				data_src.push_back(src(Rect(sim_idx_col[k], sim_idx_row[k], kWien, kWien)));
			}

			tran2d(data, "DCT",kWien);
			tran2d(data_src, "DCT", kWien);
			tran1d(data, "HAAR", kWien);
			tran1d(data_src, "HAAR", kWien);

			gen_wienFilter(data, sigma);
			weight_wien = calculate_weight_wien(data, sigma);
			wienFiltering(data_src, data, kWien);

			inv_tran_3d(data_src, "DCT", "HAAR", kWien);

			aggregation(numerator_wien, denominator_wien,
				sim_idx_row, sim_idx_col, data_src, weight_wien, kWien, kaiser);
		}
	}
	dst = numerator_wien / denominator_wien;

	return 0;
}

int BM3D(Mat& src, Mat& img_basic, Mat& dst, double sigma) {
    int Channel = src.channels();
    if (Channel == 1) {
        BM3D_gray(src, img_basic, dst, sigma);
        return 1;
    }
    else if (Channel == 3) {
        Mat src_channel[3];
        Mat basic_channel[3];
        Mat dst_channel[3];
        split(src, src_channel);
        BM3D_gray(src_channel[0], basic_channel[0], dst_channel[0], sigma);
        BM3D_gray(src_channel[1], basic_channel[1], dst_channel[1], sigma);
        BM3D_gray(src_channel[2], basic_channel[2], dst_channel[2], sigma);
        merge(basic_channel, 3, img_basic);
        merge(dst_channel, 3, dst);
        return 3;
    }
    return 0;
}

int BM3D_color_debug(Mat &src, Mat &img_basic, Mat &dst, double sigma){
    const int nHard = 39;//search area
	const int nWien = 39;
	const int kHard = 8;//patch size
	const int kWien = 8;
	const int NHard = 16;//max number
	const int NWien = 32;
	const int pHard = 3;//step
	const int pWien = 3;

	const int tao_hard = 2500;
	const int tao_wien = 400;

	const int beta = 2;
	const float lambda3d = 2.7;
	const float lambda2d = 0;

    int Height = src.rows;
	int Width = src.cols;
	int Channels = src.channels();

	//step 1 hard threshhold filtering
	if (Channels == 3)
	{
		cvtColor(src, src, CV_BGR2YUV);
	}
    else
        return -1;
    Mat channel[3];
    split(src, channel);

	vector<Mat> block_src_Y, block_src_U, block_src_V;//store the patch
	vector<int>row_idx;//patch idx along the row direction
	vector<int>col_idx;

	getPatches(channel[0], Width, Height, Channels, kHard, pHard, block_src_Y, row_idx, col_idx);
    getPatches(channel[1], kHard, block_src_U, row_idx, col_idx);
    getPatches(channel[2], kHard, block_src_V, row_idx, col_idx);

	int bn_r = row_idx.size();
	int bn_c = col_idx.size();
	tran2d(block_src_Y, "DCT",kHard);
    tran2d(block_src_U, "DCT",kHard);
    tran2d(block_src_V, "DCT",kHard);

	vector<int> sim_num;//index number for the selected similar patch in the block vector
	vector<int> sim_idx_row;//index number for the selected similar patch in the srcal Mat
	vector<int> sim_idx_col;

	vector<Mat>data_Y;//store the data during transforming and shrinking
    vector<Mat>data_U;
    vector<Mat>data_V;

	Mat kaiser = Kaiser(beta, kHard);//2-D kaiser window 
	float weight_hd[3]={1.0, 1.0, 1.0};//weights used for current relevent patch
	Mat denominator_hd_Y(src.size(), CV_32FC1,Scalar::all(0));
    Mat denominator_hd_U(src.size(), CV_32FC1,Scalar::all(0));
    Mat denominator_hd_V(src.size(), CV_32FC1,Scalar::all(0));

	Mat numerator_hd_Y(src.size(), CV_32FC1,Scalar::all(0));
    Mat numerator_hd_U(src.size(), CV_32FC1,Scalar::all(0));
    Mat numerator_hd_V(src.size(), CV_32FC1,Scalar::all(0));

    int selectedNum = 0;
	for (int i = 0; i < bn_r; i++)
	{
		for (int j = 0; j < bn_c; j++)
		{
			//for each pack in the block
			sim_num.clear();
			sim_idx_row.clear();
			sim_idx_col.clear();
			data_Y.clear();
            data_U.clear();
            data_V.clear();

            //getSimilarPatch(block_src_Y, data_Y, sim_num,
            //    i, j, bn_r, bn_c, int((nHard - kHard) / pHard) + 1, NHard, tao_hard);
            //cout << "(" << i << "," << j << ") bn_r = " << bn_r << " bn_c = " << bn_c << " blocksize = " << block_src_Y.size() << endl;

			getSimilarPatchNum(block_src_Y, sim_num, selectedNum,
				i, j, bn_r,bn_c,int((nHard-kHard) / pHard)+1, NHard,tao_hard);//block matching
           
            getSimilarPatch_withnum(block_src_Y, data_Y, sim_num, selectedNum);
            getSimilarPatch_withnum(block_src_U, data_U, sim_num, selectedNum);
            getSimilarPatch_withnum(block_src_V, data_V, sim_num, selectedNum);

			for (int k = 0; k < sim_num.size(); k++)//calculate idx in the left-top corner
			{
				sim_idx_row.push_back(row_idx[sim_num[k] / bn_c]);
				sim_idx_col.push_back(col_idx[sim_num[k] % bn_c]);
			}

			tran1d(data_Y ,"HAAR",kHard);//3-D transforming
            tran1d(data_U ,"HAAR",kHard);
            tran1d(data_V ,"HAAR",kHard);

			shrink(data_Y,lambda3d*sigma);//shrink the cofficient
            shrink(data_U,lambda3d*sigma);
            shrink(data_V,lambda3d*sigma);

			weight_hd[0] = calculate_weight_hd(data_Y,sigma);
            weight_hd[1] = calculate_weight_hd(data_Y,sigma);
            weight_hd[2] = calculate_weight_hd(data_Y,sigma);

			inv_tran_3d(data_Y, "DCT", "HAAR",kHard);//3-D inverse transforming
            inv_tran_3d(data_U, "DCT", "HAAR",kHard);
            inv_tran_3d(data_V, "DCT", "HAAR",kHard);
			
			aggregation(numerator_hd_Y, denominator_hd_Y, sim_idx_row, sim_idx_col, data_Y,weight_hd[0],kHard,kaiser);//aggregation using weigths
            aggregation(numerator_hd_U, denominator_hd_U, sim_idx_row, sim_idx_col, data_U,weight_hd[1],kHard,kaiser);//aggregation using weigths
            aggregation(numerator_hd_V, denominator_hd_V, sim_idx_row, sim_idx_col, data_V,weight_hd[2],kHard,kaiser);//aggregation using weigths
		}
	}
    
    Mat channel_basic[3];
    channel_basic[0] = numerator_hd_Y / denominator_hd_Y;
    channel_basic[1] = numerator_hd_U / denominator_hd_U;
    channel_basic[2] = numerator_hd_V / denominator_hd_V;
	merge(channel_basic, 3, img_basic);
    cvtColor(img_basic, img_basic, CV_YUV2BGR);

    waitKey(10000);
    
    cout << channel_basic[0].size() << endl;
    block_src_Y.clear();
    block_src_U.clear();
    block_src_V.clear();
	//step 2 wiena filtering
	vector<Mat> block_basic_Y, block_basic_V, block_basic_U;
	row_idx.clear();
	col_idx.clear();

	getPatches(channel_basic[0], Width, Height, Channels, kHard, pHard, block_basic_Y, row_idx, col_idx);
    getPatches(channel_basic[1], kHard, block_basic_V, row_idx, col_idx);
    getPatches(channel_basic[2], kHard, block_basic_U, row_idx, col_idx);
	bn_r = row_idx.size();
	bn_c = col_idx.size();
    cout << "bn_r = " << bn_r << " bn_c = " << bn_c << " blocksize = " << block_basic_Y.size() << endl;
	vector<Mat> data_src_Y, data_src_V, data_src_U;

    float weight_wien[3] = { 1.0, 1.0, 1.0 };//weights used for current relevent patch
	Mat denominator_wien_Y(src.size(), CV_32FC1, Scalar::all(0));
    Mat denominator_wien_V(src.size(), CV_32FC1, Scalar::all(0));
    Mat denominator_wien_U(src.size(), CV_32FC1, Scalar::all(0));
	Mat numerator_wien_Y(src.size(), CV_32FC1, Scalar::all(0));
    Mat numerator_wien_V(src.size(), CV_32FC1, Scalar::all(0));
    Mat numerator_wien_U(src.size(), CV_32FC1, Scalar::all(0));

    
    
	for (int i = 0; i < bn_r; i++)
	{
		for (int j = 0; j < bn_c; j++)
		{
			//for each pack in the basic estimate
			sim_num.clear();
			sim_idx_row.clear();
			sim_idx_col.clear();
			data_Y.clear();
			data_src_Y.clear();
            data_U.clear();
            data_src_U.clear();
            data_V.clear();
            data_src_V.clear();

			getSimilarPatchNum(block_basic_Y, sim_num, selectedNum, 
                i, j, bn_r, bn_c, int((nWien - kWien) / pWien) + 1, NWien, tao_wien);//block matching

            getSimilarPatch_withnum(block_basic_Y, data_Y, sim_num, selectedNum);
            getSimilarPatch_withnum(block_basic_U, data_U, sim_num, selectedNum);
            getSimilarPatch_withnum(block_basic_V, data_V, sim_num, selectedNum);


            cout << "(" << i << "," << j << ") bn_r = " << bn_r << " bn_c = " << bn_c << " blocksize = " << block_basic_Y.size() << endl;

			for (int k = 0; k < sim_num.size(); k++)//calculate idx in the left-top corner
			{
				sim_idx_row.push_back(row_idx[sim_num[k] / bn_c]);
				sim_idx_col.push_back(col_idx[sim_num[k] % bn_c]);
				data_src_Y.push_back(channel[0](Rect(sim_idx_col[k], sim_idx_row[k], kWien, kWien)));
                data_src_U.push_back(channel[1](Rect(sim_idx_col[k], sim_idx_row[k], kWien, kWien)));
                data_src_V.push_back(channel[2](Rect(sim_idx_col[k], sim_idx_row[k], kWien, kWien)));
			}

            cout << "(" << i << "," << j << ") data_Y = " << data_Y.size() << " data_src_Y = " << data_src_Y.size() << endl;

			tran2d(data_Y, "DCT",kWien);
			tran2d(data_src_Y, "DCT", kWien);
			tran1d(data_Y, "HAAR", kWien);
			tran1d(data_src_Y, "HAAR", kWien);

            tran2d(data_U, "DCT",kWien);
			tran2d(data_src_U, "DCT", kWien);
			tran1d(data_U, "HAAR", kWien);
			tran1d(data_src_U, "HAAR", kWien);

            tran2d(data_V, "DCT",kWien);
			tran2d(data_src_V, "DCT", kWien);
			tran1d(data_V, "HAAR", kWien);
			tran1d(data_src_V, "HAAR", kWien);

            cout << "(" << i << "," << j << ") data_Y = " << data_Y.size() << " data_src_Y = " << data_src_Y.size() << endl;

			gen_wienFilter(data_Y, sigma);
			weight_wien[0] = calculate_weight_wien(data_Y, sigma);
			wienFiltering(data_src_Y, data_Y, kWien);

            gen_wienFilter(data_U, sigma);
			weight_wien[1] = calculate_weight_wien(data_U, sigma);
			wienFiltering(data_src_U, data_U, kWien);

            gen_wienFilter(data_V, sigma);
			weight_wien[2] = calculate_weight_wien(data_V, sigma);
			wienFiltering(data_src_V, data_V, kWien);

			inv_tran_3d(data_src_Y, "DCT", "HAAR", kWien);
            inv_tran_3d(data_src_U, "DCT", "HAAR", kWien);
            inv_tran_3d(data_src_V, "DCT", "HAAR", kWien);

			aggregation(numerator_wien_Y, denominator_wien_Y,
				sim_idx_row, sim_idx_col, data_src_Y, weight_wien[0], kWien, kaiser);
            aggregation(numerator_wien_U, denominator_wien_U,
				sim_idx_row, sim_idx_col, data_src_U, weight_wien[1], kWien, kaiser);
            aggregation(numerator_wien_V, denominator_wien_V,
				sim_idx_row, sim_idx_col, data_src_V, weight_wien[2], kWien, kaiser);
		}
	}
    Mat channel_dst[3];
    channel_dst[0] = numerator_wien_Y / denominator_wien_Y;
    channel_dst[1] = numerator_wien_U / denominator_wien_U;
    channel_dst[2] = numerator_wien_V / denominator_wien_V;
	merge(channel_dst, 3, dst);

    block_basic_Y.clear();
    block_basic_U.clear();
    block_basic_V.clear();

    cvtColor(dst, dst, CV_YUV2BGR);

	return 0;
}

int run(){
    Mat img = imread("Messi.jpg");
    Mat img_gray = imread("Messi.jpg", 0);//读取方式和sigma都需要修改
    //Mat img_gray;
    //img.convertTo(img_gray, CV_BGR2GRAY);
    //cvtColor(img, img_gray, CV_BGR2GRAY);

    //Mat noise(img.size(), CV_32FC3);
    //Mat noise_gray(img_gray.size(), CV_32FC1);
    float theta = 0, sigma_orig = 3, r = 1, h = 10;
    int type = 2;
    //RNG rng;
    //rng.fill(noise, RNG::NORMAL, theta, sigma);
    //rng.fill(noise_gray, RNG::NORMAL, theta, sigma);
    Mat img_noise, img_gray_noise;
    img.convertTo(img_noise, CV_32FC3);
    img_gray.convertTo(img_gray_noise, CV_32FC1);
    
    AddNoise(img_noise, img_noise, sigma_orig, type);
    AddNoise(img_gray_noise, img_gray_noise, sigma_orig, type);
    //img_noise += noise;
    //img_gray_noise += noise_gray;

    float sigma = Standard_Deviation_Estimator(img_noise);
    float sigma_gray = Standard_Deviation_Estimator(img_gray_noise);

    //sigma = sigma_gray = sigma_orig;
    
    Mat bm3d(img.size(), CV_32FC3);
    Mat bm3d_gray(img.size(), CV_32FC1);
    Mat bm3d_basic(img.size(), CV_32FC3);
    Mat bm3d_gray_basic(img.size(), CV_32FC1);
    BM3D(img_noise, bm3d_basic, bm3d, sigma);
    BM3D(img_gray_noise, bm3d_gray_basic, bm3d_gray, sigma_gray);
    bm3d_basic.convertTo(bm3d_basic, CV_8UC3);
    bm3d.convertTo(bm3d, CV_8UC3);
    bm3d_gray_basic.convertTo(bm3d_gray_basic, CV_8UC3);
    bm3d_gray.convertTo(bm3d_gray, CV_8UC3);
    
    img_noise.convertTo(img_noise, CV_8UC3);
    img_gray_noise.convertTo(img_gray_noise, CV_8UC1);
    cout << "noisy" << endl;
    cout << "PSNR = " << PSNR(img, img_noise) << "dB" << endl;
    cout << "SSIM = " << SSIM(img, img_noise) << endl;
    cout << "noisy_gray" << endl;
    cout << "PSNR = " << PSNR(img_gray, img_gray_noise) << "dB" << endl;
    cout << "SSIM = " << SSIM(img_gray, img_gray_noise) << endl << endl;

    imwrite("img.jpg", img);
    imwrite("img_gray.jpg", img_gray);
    imwrite("img_noise.jpg", img_noise);
    imwrite("img_gray_noise.jpg", img_gray_noise);
    
    cout << "bm3d_gray" << endl;
    cout << "PSNR = " << PSNR(img_gray, bm3d_gray) << "dB" << endl;
    cout << "SSIM = " << SSIM(img_gray, bm3d_gray) << endl << endl;
    cout << "bm3d" << endl;
    cout << "PSNR = " << PSNR(img, bm3d) << "dB" << endl;
    cout << "SSIM = " << SSIM(img, bm3d) << endl << endl;
    cout << "bm3d_gray_basic" << endl;
    cout << "PSNR = " << PSNR(img_gray, bm3d_gray_basic) << "dB" << endl;
    cout << "SSIM = " << SSIM(img_gray, bm3d_gray_basic) << endl << endl;
    cout << "bm3d_basic" << endl;
    cout << "PSNR = " << PSNR(img, bm3d_basic) << "dB" << endl;
    cout << "SSIM = " << SSIM(img, bm3d_basic) << endl << endl;
    
    imwrite("bm3d.jpg", bm3d);
    imwrite("bm3d_basic.jpg", bm3d_basic);
    imwrite("bm3d_gray.jpg", bm3d_gray);
    imwrite("bm3d_gray_basic.jpg", bm3d_gray_basic);
    
    
    Mat local_gaussian(img.size(), CV_8UC3);
    Mat local_gaussian_gray(img.size(), CV_8UC1);
    Local_Mean_filter(img_noise, local_gaussian);
    Local_Mean_filter(img_gray_noise, local_gaussian_gray);

    Mat local_average(img.size(), CV_8UC3);
    Mat local_average_gray(img.size(), CV_8UC1);
    Local_Mean_filter(img_noise, local_average, 0);
    Local_Mean_filter(img_gray_noise, local_average_gray, 0);
    cout << "local_average_gray" << endl;
    cout << "PSNR = " << PSNR(img_gray, local_average_gray) << "dB" << endl;
    cout << "SSIM = " << SSIM(img_gray, local_average_gray) << endl << endl;
    cout << "local_average" << endl;
    cout << "PSNR = " << PSNR(img, local_average) << "dB" << endl;
    cout << "SSIM = " << SSIM(img, local_average) << endl << endl;
    cout << "local_gaussian_gray" << endl;
    cout << "PSNR = " << PSNR(img_gray, local_gaussian_gray) << "dB" << endl;
    cout << "SSIM = " << SSIM(img_gray, local_gaussian_gray) << endl << endl;
    cout << "local_gaussian" << endl;
    cout << "PSNR = " << PSNR(img, local_gaussian) << "dB" << endl;
    cout << "SSIM = " << SSIM(img, local_gaussian) << endl << endl;

    imwrite("local_gaussian.jpg", local_gaussian);
    imwrite("local_average.jpg", local_average);
    imwrite("local_gaussian_gray.jpg", local_gaussian_gray);
    imwrite("local_average_gray.jpg", local_average_gray);

    Mat adapt_sigma(img.size(), CV_8UC3);
    Mat adapt_sigma_gray(img.size(), CV_8UC1);
    Adaptive_Mean_Filter(img_noise, adapt_sigma, sigma, r, 0);
    Adaptive_Mean_Filter(img_gray_noise, adapt_sigma_gray, sigma_gray, r, 0);
    Mat adapt_mid(img.size(), CV_8UC3);
    Mat adapt_mid_gray(img.size(), CV_8UC1);
    Adaptive_Mean_Filter(img_noise, adapt_mid, sigma, r, 1);
    Adaptive_Mean_Filter(img_gray_noise, adapt_mid_gray, sigma_gray, r, 1);
    cout << "adapt_sigma_gray" << endl;
    cout << "PSNR = " << PSNR(img_gray, adapt_sigma_gray) << "dB" << endl;
    cout << "SSIM = " << SSIM(img_gray, adapt_sigma_gray) << endl << endl;
    cout << "adapt_sigma" << endl;
    cout << "PSNR = " << PSNR(img, adapt_sigma) << "dB" << endl;
    cout << "SSIM = " << SSIM(img, adapt_sigma) << endl << endl;
    cout << "adapt_mid_gray" << endl;
    cout << "PSNR = " << PSNR(img_gray, adapt_mid_gray) << "dB" << endl;
    cout << "SSIM = " << SSIM(img_gray, adapt_mid_gray) << endl << endl;
    cout << "adapt_mid" << endl;
    cout << "PSNR = " << PSNR(img, adapt_mid) << "dB" << endl;
    cout << "SSIM = " << SSIM(img, adapt_mid) << endl << endl;

    imwrite("adapt_sigma.jpg", adapt_sigma);
    imwrite("adapt_mid.jpg", adapt_mid);
    imwrite("adapt_sigma_gray.jpg", adapt_sigma_gray);
    imwrite("adapt_mid_gray.jpg", adapt_mid_gray);
    
    Mat nlm_gaussian(img.size(), CV_8UC3);
    Mat nlm_gaussian_gray(img.size(), CV_8UC1);
    NLM_withsigma(img_noise, nlm_gaussian, sigma, h);
    NLM_withsigma(img_gray_noise, nlm_gaussian_gray, sigma_gray, h);
    Mat nlm_average(img.size(), CV_8UC3);
    Mat nlm_average_gray(img.size(), CV_8UC1);
    NLM_withsigma(img_noise, nlm_average, sigma, h, 0);
    NLM_withsigma(img_gray_noise, nlm_average_gray, sigma_gray, h, 0);
    cout << "nlm_average_gray" << endl;
    cout << "PSNR = " << PSNR(img_gray, nlm_average_gray) << "dB" << endl;
    cout << "SSIM = " << SSIM(img_gray, nlm_average_gray) << endl << endl;
    cout << "nlm_average" << endl;
    cout << "PSNR = " << PSNR(img, nlm_average) << "dB" << endl;
    cout << "SSIM = " << SSIM(img, nlm_average) << endl << endl;
    cout << "nlm_gaussian_gray" << endl;
    cout << "PSNR = " << PSNR(img_gray, nlm_gaussian_gray) << "dB" << endl;
    cout << "SSIM = " << SSIM(img_gray, nlm_gaussian_gray) << endl << endl;
    cout << "nlm_gaussian" << endl;
    cout << "PSNR = " << PSNR(img, nlm_gaussian) << "dB" << endl;
    cout << "SSIM = " << SSIM(img, nlm_gaussian) << endl << endl;

    imwrite("nlm_gaussian.jpg", nlm_gaussian);
    imwrite("nlm_average.jpg", nlm_average);
    imwrite("nlm_gaussian_gray.jpg", nlm_gaussian_gray);
    imwrite("nlm_average_gray.jpg", nlm_average_gray);
    /*
    imwrite("img.jpg",img);
    imwrite("img_gray.jpg",img_gray);
    imwrite("img_noise.jpg",img_noise);
    imwrite("img_gray_noise.jpg",img_gray_noise);
    
    imwrite("bm3d.jpg", bm3d);
    imwrite("bm3d_basic.jpg", bm3d_basic);
    imwrite("bm3d_gray.jpg", bm3d_gray);
    imwrite("bm3d_gray_basic.jpg", bm3d_gray_basic);
    
    imwrite("local_gaussian.jpg", local_gaussian);
    imwrite("local_average.jpg", local_average);
    imwrite("local_gaussian_gray.jpg", local_gaussian_gray);
    imwrite("local_average_gray.jpg", local_average_gray);

    imwrite("adapt_sigma.jpg", adapt_sigma);
    imwrite("adapt_mid.jpg", adapt_mid);
    imwrite("adapt_sigma_gray.jpg", adapt_sigma_gray);
    imwrite("adapt_mid_gray.jpg", adapt_mid_gray);
    
    imwrite("nlm_gaussian.jpg", nlm_gaussian);
    imwrite("nlm_average.jpg", nlm_average);
    imwrite("nlm_gaussian_gray.jpg", nlm_gaussian_gray);
    imwrite("nlm_average_gray.jpg", nlm_average_gray);
    */
    return 0;
}

int main() {
    Mat src = imread("Messi.jpg", 0);
    Mat dst;
    src.convertTo(src, CV_32FC1);
    //cvtColor(src, dst, COLOR_GRAY2BGR);
    AddNoise(src, dst, 25, 1);
    //src.convertTo(dst, COLOR_GRAY2BGR);
    //src.convertTo(src, COLOR_GRAY2BGR);
    cout << dst.channels() << endl;
    dst.convertTo(dst, CV_8UC1);

    namedWindow("src",CV_WINDOW_AUTOSIZE);
    imshow("src", dst);
    waitKey(10000);
    return 0;
    //run();
    //return 0;
    /*
    Mat src = imread("Messi.jpg");
    //   Mat src = imread("")
    Mat dst;
    src.copyTo(dst);
    int minSize = 3; // 滤波器窗口的起始尺寸
    int maxSize = 7; // 滤波器窗口的最大尺寸
    AddNoise(src, dst, 3, 2);
    //addImpulseNoise(dst, 10000);
    Mat noisy = dst.clone();
    Mat img(noisy.size(), noisy.type());
    medianBlur(noisy, img, 5);
    //copyMakeBorder(src, dst, (maxSize - 1) / 2, (maxSize - 1) / 2, (maxSize - 1) / 2, (maxSize - 1) / 2, BorderTypes::BORDER_REFLECT);
    copyMakeBorder(dst, dst, (maxSize - 1) / 2, (maxSize - 1) / 2, (maxSize - 1) / 2, (maxSize - 1) / 2, BORDER_REFLECT);
    vector<Mat> channels_dst;
    //vector<Mat> channels_src;
    split(dst, channels_dst);
    //split(dst, channels_src);
    adaptiveMediaFilter(dst, dst, maxSize, minSize);
    for (int i = 0; i < 3; ++i)
    {
        adaptiveMediaFilter(channels_dst[i], channels_dst[i], maxSize, minSize);
    }
    merge(channels_dst, dst);
    //merge(channels_src, dst);
    Rect r = Rect((maxSize - 1) / 2, (maxSize - 1) / 2, dst.cols - (maxSize - 1), dst.rows - (maxSize - 1));
    Mat res = dst(r);
    //   adaptiveMediaFilter(src, dst);
 //   imwrite("audreybef.jpg", src);
 //   imwrite("audreyaft.jpg", dst);
    cout << "noisy" << endl;
    cout << "PSNR = " << PSNR(src, noisy) << "dB" << endl;
    cout << "SSIM = " << SSIM(src, noisy) << endl << endl;
    cout << "mid" << endl;
    cout << "PSNR = " << PSNR(src, img) << "dB" << endl;
    cout << "SSIM = " << SSIM(src, img) << endl << endl;
    cout << "a_mid" << endl;
    cout << "PSNR = " << PSNR(src, res) << "dB" << endl;
    cout << "SSIM = " << SSIM(src, res) << endl << endl;
    //cout << "noisy" << endl;
    //cout << "PSNR = " << PSNR(src, res) << "dB" << endl;
    //cout << "SSIM = " << SSIM(src, res) << endl << endl;
    imwrite("impulse_noisy.jpg", noisy);
    imwrite("mid.jpg", img);
    imwrite("a_mid.jpg", res);
    waitKey(100000);
    return 0;
    
    //Mat img = imread("Messi.jpg");
    img.convertTo(img, CV_32FC3);
    //Mat noisy(img.size(), img.type());
    AddNoise(img, noisy, 3, 2);
    noisy.convertTo(noisy, CV_8UC3);
    imshow("img",noisy);
    waitKey(10000);
    */
    //Mat img = imread("Messi.jpg");
    //Mat img_gray = imread("Messi.jpg", 0);

    //Mat img_noise, img_gray_noise;
    //img.convertTo(img_noise, CV_32FC3);
    //img_gray.convertTo(img_gray_noise, CV_32FC1);
    //int sigma = 50;
    //int type = 4;
    //AddNoise(img_noise, img_noise, sigma, type);
    //AddNoise(img_gray_noise, img_gray_noise, sigma, type);
    //img_noise.convertTo(img_noise, CV_8UC3);
    //img_gray_noise.convertTo(img_gray_noise, CV_8UC1);
    //namedWindow("img_noise", CV_WINDOW_AUTOSIZE);
    //imshow("img_noise", img_noise);
    //namedWindow("img_gray_noise", CV_WINDOW_AUTOSIZE);
    //imshow("img_gray_noise", img_gray_noise);
  
    //waitKey(5000);
    //Mat img = imread("Messi.jpg");
    //img = img(Rect(100, 0, 50, 50));
    //imwrite("arm.jpg", img);
    /*
    Mat src = imread("Messi.jpg");
    //   Mat src = imread("")
    Mat dst;
    src.copyTo(dst);
    int minSize = 3; // 滤波器窗口的起始尺寸
    int maxSize = 7; // 滤波器窗口的最大尺寸
    addImpulseNoise(dst, 10000);
    Mat noisy = dst.clone();
    Mat img(noisy.size(), noisy.type());
    medianBlur(noisy, img, 5);
    //copyMakeBorder(src, dst, (maxSize - 1) / 2, (maxSize - 1) / 2, (maxSize - 1) / 2, (maxSize - 1) / 2, BorderTypes::BORDER_REFLECT);
    copyMakeBorder(dst, dst, (maxSize - 1) / 2, (maxSize - 1) / 2, (maxSize - 1) / 2, (maxSize - 1) / 2, BORDER_REFLECT);
    vector<Mat> channels_dst;
    vector<Mat> channels_src;
    split(dst, channels_dst);
    //split(dst, channels_src);
    for (int i = 0; i < 3; ++i)
    {
        adaptiveMediaFilter(channels_dst[i], channels_dst[i], maxSize, minSize);
    }
    merge(channels_dst, dst);
    //merge(channels_src, dst);
    Rect r = Rect((maxSize - 1) / 2, (maxSize - 1) / 2, dst.cols - (maxSize - 1), dst.rows - (maxSize - 1));
    Mat res = dst(r);
    //   adaptiveMediaFilter(src, dst);
 //   imwrite("audreybef.jpg", src);
 //   imwrite("audreyaft.jpg", dst);
    cout << "noisy" << endl;
    cout << "PSNR = " << PSNR(src, noisy) << "dB" << endl;
    cout << "SSIM = " << SSIM(src, noisy) << endl << endl;
    cout << "res" << endl;
    cout << "PSNR = " << PSNR(src, img) << "dB" << endl;
    cout << "SSIM = " << SSIM(src, img) << endl << endl;
    //cout << "noisy" << endl;
    //cout << "PSNR = " << PSNR(src, res) << "dB" << endl;
    //cout << "SSIM = " << SSIM(src, res) << endl << endl;
    imwrite("impulse_noisy.jpg", noisy);
    imwrite("mid.jpg", img);
    imwrite("a_mid.jpg", res);
    waitKey(100000);
    return 0;
    */
    /*BM3D_gray测试——灰度图像*/
    /*
    Mat img_gray = imread("house.png", 0);

    Mat noise = Mat::zeros(img_gray.rows, img_gray.cols, CV_32FC1);
    double theta = 0, sigma = 25, r = 1;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);

    img_gray.convertTo(img_gray, CV_32FC1);
    Mat img_gray_noise = img_gray + noise;
    //img_gray_noise.convertTo(img_gray_noise, CV_8UC1);

    Mat img_basic(img_gray.rows, img_gray.cols, CV_32FC1);
    Mat dst(img_gray.rows, img_gray.cols, CV_32FC1);

    BM3D_gray(img_gray_noise, img_basic, dst, sigma);

    img_gray_noise.convertTo(img_gray_noise, CV_8UC1);
    namedWindow("img_gray_noise", CV_WINDOW_AUTOSIZE);
    imshow("img_gray_noise", img_gray_noise);

    img_basic.convertTo(img_basic, CV_8UC1);
    namedWindow("img_basic", CV_WINDOW_AUTOSIZE);
    imshow("img_basic", img_basic);

    dst.convertTo(dst, CV_8UC1);
    namedWindow("dst", CV_WINDOW_AUTOSIZE);
    imshow("dst", dst);

    waitKey(10000);
    return 0;
    */
    /*BM3D_gray测试——彩色图像*/
    /*
    Mat img = imread("Messi.jpg");

    Mat noise = Mat::zeros(img.rows, img.cols, CV_32FC3);
    double theta = 0, sigma = 25, r = 1;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);

    img.convertTo(img, CV_32FC3);
    Mat img_noise = img + noise;
    //img_gray_noise.convertTo(img_gray_noise, CV_8UC1);

    Mat img_basic(img.rows, img.cols, CV_32FC3);
    Mat dst(img.rows, img.cols, CV_32FC3);

    BM3D(img_noise, img_basic, dst, sigma);

    img_noise.convertTo(img_noise, CV_8UC3);
    namedWindow("img_gray_noise", CV_WINDOW_AUTOSIZE);
    imshow("img_gray_noise", img_noise);

    img_basic.convertTo(img_basic, CV_8UC3);
    namedWindow("img_basic", CV_WINDOW_AUTOSIZE);
    imshow("img_basic", img_basic);

    dst.convertTo(dst, CV_8UC3);
    namedWindow("dst", CV_WINDOW_AUTOSIZE);
    imshow("dst", dst);

    waitKey(10000000);
    return 0;
    */
    /*
    Mat img = imread("Hofn.jpg");
    Mat img_gray;
    //Mat dst;

    
    cout << "size = " << img.size << endl;
    cout << "channel = " << img.channels() << endl;
    cout << "type = " << img.type() << endl;
    
    cvtColor(img, img_gray, CV_BGR2GRAY);

    Mat noise = Mat::zeros(img.rows, img.cols, CV_64FC1);
    double theta = 0, sigma = 10, r = 1;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);

    img_gray.convertTo(img_gray, CV_64FC1);
    Mat img_gray_noise = img_gray + noise;
    img_gray_noise.convertTo(img_gray_noise, CV_8UC1);

    Mat dst = Mat::zeros(img_gray.rows, img_gray.cols, CV_8UC1);

    Adaptive_Mean_Filter(img_gray_noise, dst, sigma, r, 1);
    
    cout << "PSNR = " << PSNR(img_gray_noise, dst) << "dB" << endl;
    cout << "SSIM = " << SSIM(dst, img_gray_noise) << endl;
    namedWindow("img_gray_noise", CV_WINDOW_AUTOSIZE);
    imshow("img_gray_noise", img_gray_noise);

    namedWindow("dst", CV_WINDOW_AUTOSIZE);
    imshow("dst", dst);

    waitKey(10000);
    return 0;
    */
    /*自适应NLM去噪 灰度图像*/
    /*
    cvtColor(img, img_gray, CV_BGR2GRAY);
    
    Mat noise = Mat::zeros(img.rows, img.cols, CV_64FC1);
    double theta = 0, sigma = 10;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);

    img_gray.convertTo(img_gray, CV_64FC1);
    Mat img_gray_noise = img_gray + noise;
    img_gray_noise.convertTo(img_gray_noise, CV_8UC1);

    double estimate = 0;
    estimate = Standard_Deviation_Estimator(img_gray_noise);
    cout << "srcal sigma = " << sigma << endl;
    cout << "estimated sigma = " << estimate << endl;

    
    Mat dst = Mat::zeros(img_gray.rows,img_gray.cols, CV_8UC1);
    //cout << "img_gray.type = " << img_gray.type() << endl;
    //cout << CV_8UC1 << endl;

    double t = 0;
    t = (double)getTickCount();
    NLM_withsigma(img_gray_noise, dst, estimate);
    t = ((double)getTickCount() - t) / (double)getTickFrequency();
    
    namedWindow("img_gray_noise",CV_WINDOW_AUTOSIZE);
    imshow("img_gray_noise", img_gray_noise);

    namedWindow("dst", CV_WINDOW_AUTOSIZE);
    imshow("dst", dst);

    cout << "time = " << t << "s" << endl;
    
    waitKey(1000000);
    return 0;
    */

    /*灰度图像方差预测*/
    /*
    Mat noise = Mat::zeros(img.rows, img.cols, CV_64FC1);
    double theta = 0, sigma = 10, h = 10;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);

    img_gray.convertTo(img_gray, CV_64FC1);
    Mat img_gray_noise = img_gray + noise;
    img_gray_noise.convertTo(img_gray_noise, CV_8UC1);
   
    double estimate=0;
    estimate = Standard_Deviation_Estimator(img_gray_noise);
    cout << "srcal sigma = " << sigma << endl;
    cout << "estimated sigma = " << estimate << endl;

    return 0;
    */
    /*彩色图像方差测试*/
    /*
    Mat noise = Mat::zeros(img.rows, img.cols, CV_64FC3);
    double theta = 0, sigma = 10, h = 10;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);

    img.convertTo(img, CV_64FC3);
    Mat img_noise = img + noise;
    img_noise.convertTo(img_noise, CV_8UC3);
    
    cout << "size = " << img_noise.size << endl;
    cout << "channel = " << img_noise.channels() << endl;
    cout << "type = " << img_noise.type() << endl;

    dst = Mat::zeros(img_gray.rows, img_gray.cols, img.type());
    Local_Mean_filter(img_noise, dst);

    double estimate = 0;
    estimate = Standard_Deviation_Estimator(img_noise);
    cout << "srcal sigma = " << sigma << endl;
    cout << "estimated sigma = " << estimate << endl;

    return 0;
    */


    /*求导法寻找匀色区域 失败*/
    /*
    namedWindow("原图", CV_WINDOW_AUTOSIZE);
    imshow("原图", img_gray);
    
    Mat dst_x = Mat::zeros(img_gray.rows, img_gray.cols, img_gray.type());
    Mat dst_y = Mat::zeros(img_gray.rows, img_gray.cols, img_gray.type());
    
    Laplacian(img_gray, dst, img.depth());
    //Derivation(img_gray, dst_x, 1, 0);
    //Derivation(img_gray, dst_y, 0, 1);
    //dst = dst_x + dst_y;

    namedWindow("未加噪，未平滑，未阈值处理", CV_WINDOW_AUTOSIZE);
    imshow("未加噪，未平滑，未阈值处理", dst);

    threshold(dst, dst, 50, 255, THRESH_BINARY);
    //namedWindow("未加噪，未平滑，阈值处理", CV_WINDOW_AUTOSIZE);
    //imshow("未加噪，未平滑，阈值处理", dst);

    Local_Mean_filter(img_gray, img_gray, 5, 0);
    Laplacian(img_gray, dst, img.depth());
    //Derivation(img_gray, dst_x, 1, 0);
    //Derivation(img_gray, dst_y, 0, 1);
    //dst = dst_x + dst_y;
    namedWindow("未加噪，平滑，未阈值处理", CV_WINDOW_AUTOSIZE);
    imshow("未加噪，平滑，未阈值处理", dst);

    threshold(dst, dst, 20, 255, THRESH_BINARY);
    //namedWindow("未加噪，平滑，阈值处理", CV_WINDOW_AUTOSIZE);
    //imshow("未加噪，平滑，阈值处理", dst);

    Mat noise = Mat::zeros(img.rows, img.cols, CV_64FC1);
    double theta = 0, sigma = 10, h = 10;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);

    img_gray.convertTo(img_gray, CV_64FC1);
    Mat img_gray_noise = img_gray + noise;
    img_gray_noise.convertTo(img_gray_noise, CV_8UC1);

    //namedWindow("加噪", CV_WINDOW_AUTOSIZE);
    //imshow("加噪", img_gray_noise);

    //Derivation(img_gray_noise, dst_x, 1, 0);
    //Derivation(img_gray_noise, dst_y, 0, 1);
    //dst = dst_x + dst_y;
    Laplacian(img_gray_noise, dst, img.depth());

    namedWindow("加噪，未平滑，未阈值处理", CV_WINDOW_AUTOSIZE);
    imshow("加噪，未平滑，未阈值处理", dst);

    threshold(dst, dst, 50, 255, THRESH_BINARY);
    //namedWindow("加噪，未平滑，阈值处理", CV_WINDOW_AUTOSIZE);
    //imshow("加噪，未平滑，阈值处理", dst);

    Local_Mean_filter(img_gray_noise, img_gray_noise, 5, 0);

    //namedWindow("加噪平滑", CV_WINDOW_AUTOSIZE);
    //imshow("加噪平滑", img_gray_noise);

    //Derivation(img_gray_noise, dst_x, 1, 0);
    //Derivation(img_gray_noise, dst_y, 0, 1);
    //dst = dst_x + dst_y;
    Laplacian(img_gray_noise, dst, img.depth());
    namedWindow("加噪，平滑，未阈值处理", CV_WINDOW_AUTOSIZE);
    imshow("加噪，平滑，未阈值处理", dst);

    threshold(dst, dst, 20, 255, THRESH_BINARY);
    //namedWindow("加噪，平滑，阈值处理", CV_WINDOW_AUTOSIZE);
    //imshow("加噪，平滑，阈值处理", dst);
    */

    /*******彩色图像测试 Mean and Standard_Deviation******/
    /*
    Mat img_area = img(Rect(300, 400, 60, 60));
    
    Mat m, sd;
    meanStdDev(img_area, m, sd);
    cout << "srcal:" << endl;
    cout << m << endl;
    cout << sd << endl;

    Mat noise = Mat::zeros(img.rows, img.cols, CV_64FC3);
    double theta = 0, sigma = 10, h = 10;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);
    
    meanStdDev(noise, m, sd);
    cout << "noise:" << endl;
    cout << m << endl;
    cout << sd << endl;
    img.convertTo(img, CV_64FC3);
    Mat img_noise = img + noise;

    Mat img_noise_area = img_noise(Rect(300, 400, 60, 60));

    meanStdDev(img_noise_area, m, sd);
    cout << "src:" << endl;
    cout << m << endl;
    cout << sd << endl;

    img.convertTo(img,CV_8UC3);
    Mat img_la;
    Mat img_so, img_so_x, img_so_y;
    Mat img_sc, img_sc_x, img_sc_y;
    //cout << img.depth() << endl;
   
    //cvtColor(img, img_gray, COLOR_BGR2GRAY);

    Laplacian(img, img_la, img.depth());
    Sobel(img, img_so_x, img.depth(), 1, 0);
    Sobel(img, img_so_y, img.depth(), 0, 1);
    img_so = img_so_x + img_so_y;
    Scharr(img, img_sc_x, img.depth(), 1, 0);
    Scharr(img, img_sc_y, img.depth(), 0, 1);
    img_sc = img_sc_x + img_sc_y;
    
    namedWindow("img_la", CV_WINDOW_AUTOSIZE);
    imshow("img_la", img_la);
    namedWindow("img_so", CV_WINDOW_AUTOSIZE);
    imshow("img_so", img_so);
    namedWindow("img_sc", CV_WINDOW_AUTOSIZE);
    imshow("img_sc", img_sc);
    
    //namedWindow("area");
    //imshow("area", img_area);
    //namedWindow("area_noise");
    //imshow("area_noise", img_noise_area);

    
    //GaussianBlur(img_area, img_area, Size(3, 3), 10);
    

    //mean;
    //meanStdDev();
    */

    /*******灰度图像测试 Gaussian & Average*******/
    /*
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    
    Mat noise = Mat::zeros(img_gray.rows, img_gray.cols, img_gray.type());
    double theta = 10, sigma = 20;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);
    Mat img_gray_noise = img_gray + noise;

    Mat dst = Mat::zeros(img_gray.rows,img_gray.cols,img_gray.type());
    //Average_Mean_filter(img_gray_noise, dst, 1);
    Gaussian_Mean_filter(img_gray_noise, dst, 1);
    namedWindow("img_gray_noise");
    imshow("img_gray_noise", img_gray_noise);
    namedWindow("dst");
    imshow("dst", dst);
    waitKey(5000);
    */
    /*
    //img_gray = Mat(4, 4, CV_8UC1);
    //cout << img_gray.channels() << endl;
    //img_gray.setTo(200);
    Mat dst = Mat::zeros(img_gray.rows, img_gray.cols, img_gray.type());
    Average_Mean_filter(img_gray, dst, 1);
    cout << "img_gray = \n" << img_gray << endl;
    cout << "dst = \n" << dst << endl;
    */

    /*******灰度图像测试 NLM*******/
    /*
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    Mat noise = Mat::zeros(img_gray.rows, img_gray.cols, img_gray.type());
    double theta = 10, sigma = 20, h = 10;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);
    Mat img_gray_noise = img_gray + noise;

    Mat dst_color = Mat::zeros(img_gray.rows, img_gray.cols, img_gray.type());
    Mat dst_gray = Mat::zeros(img_gray.rows, img_gray.cols, img_gray.type());
    double t_color, t_gray;

    t_color = (double)getTickCount();
    NLM_withsigma_color(img_gray_noise, dst_color, sigma, h);
    t_color = ((double)getTickCount() - t_color) / (double)getTickFrequency();

    t_gray = (double)getTickCount();
    NLM_withsigma(img_gray_noise, dst_gray, sigma, h);
    t_gray = ((double)getTickCount() - t_gray) / (double)getTickFrequency();

    cout << "NLM_color=" << t_color << "s" << endl;
    cout << "NLM_gray=" << t_gray << "s" << endl;

    namedWindow("img_gray_noise");
    imshow("img_gray_noise", img_gray_noise);

    namedWindow("dst_color");
    imshow("dst_color", dst_color);
 
    namedWindow("dst_gray");
    imshow("dst_gray", dst_gray);
    */

    /*******彩色图像测试 Gaussian & Average*******/
    /*
    Mat noise = Mat::zeros(img.rows, img.cols, img.type());
    double theta = 10, sigma = 20, h = 10, r = 1;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);
    Mat img_noise = img + noise;
    
    Mat dst = Mat::zeros(img.rows, img.cols, img.type());
    Local_Mean_filter(img_noise, dst, 1);

    namedWindow("img_noise");
    imshow("img_noise", img_noise);

    namedWindow("dst");
    imshow("dst", dst);
    */
    /*
    Mat dst = img(Rect(0, 0, 4, 4));
    cvtColor(dst, img_gray, COLOR_BGR2GRAY);
    uchar *data_dst = dst.data;
    cout << dst << endl;
    cout << dst.step[0] << endl;
    cout << dst.step[1] << endl;

    cout << img_gray << endl;
    cout << img_gray.step[0] << endl;
    cout << img_gray.step[1] << endl;
    */

    /*******彩色图像测试 NLM*******/
    /*
    Mat noise = Mat::zeros(img.rows, img.cols, img.type());
    double theta = 10, sigma = 20, h = 10;
    RNG rng;
    rng.fill(noise, RNG::NORMAL, theta, sigma);
    Mat img_noise = img + noise;

    Mat dst_average = Mat::zeros(img.rows, img.cols, img.type());
    Mat dst_gaussian = Mat::zeros(img.rows, img.cols, img.type());
    double t_average, t_gaussian;

    t_average = (double)getTickCount();
    NLM_withsigma_color(img_noise, dst_average, sigma, h, 0);
    t_average = ((double)getTickCount() - t_average) / (double)getTickFrequency();
    
    t_gaussian = (double)getTickCount();
    NLM_withsigma_color(img_noise, dst_gaussian, sigma, h, 1);
    t_gaussian = ((double)getTickCount() - t_gaussian) / (double)getTickFrequency();

    cout << "NLM_average=" << t_average << "s" << endl;
    cout << "NLM_gaussian=" << t_gaussian << "s" << endl;

    imwrite("img_noise.jpg", img_noise);
    imwrite("dst_average.jpg", dst_average);
    imwrite("dst_gaussian.jpg", dst_gaussian);

    namedWindow("img_gray_noise");
    imshow("img_noise", img_noise);

    namedWindow("dst_average");
    imshow("dst_average", dst_average);

    namedWindow("dst_gaussian");
    imshow("dst_gaussian", dst_gaussian);
    */
    
    waitKey(1000000);
    return 0;
}


double Standard_Deviation_Estimator_debug(Mat& src, int type = 0) {
    int channel = src.channels();

    double variance = 0;
    double standard_deviation = 0;
    int row = src.rows, col = src.cols;
    //uchar *dat = src.data;
    int kernel[9] = { 1, -2, 1, -2, 4, -2, 1, -2, 1 };

    int cal = 0;
    long long int total = 0;
    if (type == 0) {
        for (int y = 0; y < row - 2; y++) {
            for (int x = 0; x < col - 2; x++) {
                for (int z = 0; z < channel; z++) {
                    cal = 0;
                    for (int j = 0; j < 3; j++) {
                        for (int i = 0; i < 3; i++) {
                            cal += kernel[j * 3 + i] * (int)(*(src.data + (y + j) * src.step[0] + (x + i) * src.step[1] + z));
                        }
                    }
                    total += cal > 0 ? cal : -cal;
                }
            }
        }
        total /= channel;
        standard_deviation = total * (sqrt(Pi / 2)) / 6 / (row - 2) / (col - 2);
    }

    else {
        for (int y = 0; y < row - 2; y++) {
            for (int x = 0; x < col - 2; x++) {
                for (int z = 0; z < channel; z++) {
                    cal = 0;
                    for (int j = 0; j < 3; j++) {
                        for (int i = 0; i < 3; i++) {
                            cal += kernel[j * 3 + i] * (int)(*(src.data + (y + j) * src.step[0] + (x + i) * src.step[1] + z));
                        }
                    }
                    total += cal * cal;
                }
            }
        }
        variance = total / 36 / (row - 2) / (col - 2);
        variance /= channel;
    }
    if (type == 0)
        return standard_deviation;
    else
        return variance;
}

    //不成功，待开发
int fast_NLM(Mat& src_i, Mat& dst, double sigma, double h = 5, int r = 3, int R = 10) {
    int d = 2 * r + 1, D = 2 * R + 1;
    double sigma2 = sigma * sigma, h2 = h * h;
    Mat src;
    copyMakeBorder(src_i, src, R + r, R + r, R + r, R + r, BORDER_REFLECT);

    Mat kernel;
    //kernel = kernel.ones(2 * r + 1, 2 * r + 1, CV_64F) / ((2*r+1)*(2*r+1));
    kernel = getGaussianKernel(d, 1) * getGaussianKernel(d, 1).t();

    Mat area_search, area_template, area_tmp;
    int row_dst = src_i.rows, col_dst = src_i.cols, row_src = src.rows, col_src = src.cols;
    int x_dst, y_dst, x_src, y_src, x_min, x_max, y_min, y_max;
    int x_search, y_search;
    double w, w_center, w_total;
    double d_eu, cal;
    double average;
    float rate;

    int* SSI = (int*)malloc(sizeof(int) * row_src * col_src);
    for (int j = 0; j < row_src; j++) {
        for (int i = 0; i < col_src; i++) {
            SSI[j * col_src + i] = (int)src.at<uchar>(j, i) * (int)src.at<uchar>(j, i);
        }
    }

    int cal_template;
    int x_tmp, y_tmp;
    int x_template, y_template;
    for (y_dst = 0; y_dst < row_dst; y_dst++) {
        if (y_dst % 10 == 0) { rate = (float)y_dst / (float)row_dst * 100; cout << rate << "%" << endl; }
        for (x_dst = 0; x_dst < col_dst; x_dst++) {
            //area_template = src(Rect(x_dst + R, y_dst + R, d, d));//起始坐标待核实
            //area_search = src(Rect(x_dst, y_dst, D + d - 1, D + d - 1));
            x_template = x_dst + R + r;
            y_template = y_dst + R + r;
            cal_template = SSI[y_template * col_src + x_template];
            //初始化待处理
            w = 0;
            w_center = 0;
            w_total = 0;
            average = 0;
            //
            for (y_search = 0; y_search < D; y_search++) {
                for (x_search = 0; x_search < D; x_search++) {
                    //area_tmp = area_search(Rect(x_search, y_search, d, d));
                    //计算欧式距离
                    d_eu = 0; cal = d * d * cal_template;//cal需要初始化么
                    for (int j = 0; j < d; j++) {
                        for (int i = 0; i < d; i++) {
                            if ((x_search + i == R + r) && (y_search + j == R + r)) continue;
                            x_tmp = x_dst + x_search + i;
                            y_tmp = y_dst + y_search + j;
                            cal += SSI[y_tmp * col_src + x_tmp];
                            cal -= 2 * (int)src.at<uchar>(y_tmp, x_tmp) * (int)src.at<uchar>(y_template, x_template);
                            d_eu += cal * kernel.at<double>(j, i);
                        }
                    }
                    //
                    d_eu = (d_eu - 2 * sigma2 > 0) ? d_eu : 0;
                    w = exp(-d_eu / (h * h));
                    w_center = w > w_center ? w : w_center;
                    w_total += w;
                    average += w * (int)src.at<uchar>(y_dst + r, x_dst + r);
                }
            }
            w_total += w_center;
            average += w_center * (int)src.at<uchar>(y_template, x_template);//area_template(r,r)可能也行
            if (w_total > 0)
                dst.at<uchar>(y_dst, x_dst) = (uchar)(average / w_total);
            else
                dst.at<uchar>(y_dst, x_dst) = src.at<uchar>(y_template, x_template);
        }
    }
    return 0;
}

//已成功；at法 截取法 速度慢；仅处理单通道图像
int NLM(Mat& src_i, Mat& dst, int h = 5, int r = 3, int R = 10) {
    int d = 2 * r + 1, D = 2 * R + 1;
    Mat src;
    copyMakeBorder(src_i, src, R + r, R + r, R + r, R + r, BORDER_REFLECT);
    Mat kernel;
    //kernel = kernel.ones(2 * r + 1, 2 * r + 1, CV_64F) / ((2*r+1)*(2*r+1));

    kernel = getGaussianKernel(d, 1) * getGaussianKernel(d, 1).t();
    /*
    cout << kernel << endl;
    double to = 0;
    for (int i = 0; i < kernel.rows; i++) {
        for (int j = 0; j < kernel.cols; j++) {
            to += kernel.at<double>(i, j);
        }
    }
    cout << "kerneltotal=" << to << endl;
    */
    Mat area_search, area_template, area_tmp;
    int row_dst = src_i.rows, col_dst = src_i.cols, row_src = src.rows, col_src = src.cols;
    int x_dst, y_dst, x_src, y_src, x_min, x_max, y_min, y_max;
    int x_search, y_search;
    double w, w_center, w_total;
    double d_eu, cal;
    double average;
    float rate;
    for (y_dst = 0; y_dst < row_dst; y_dst++) {
        if (y_dst % 10 == 0) { rate = (float)y_dst / (float)row_dst * 100; cout << rate << "%" << endl; }
        for (x_dst = 0; x_dst < col_dst; x_dst++) {
            area_template = src(Rect(x_dst + R, y_dst + R, d, d));//起始坐标待核实
            area_search = src(Rect(x_dst, y_dst, D + d - 1, D + d - 1));
            uchar m = area_template.at<uchar>(r, r), n = area_search.at<uchar>(R + r, R + r);
            //初始化待处理
            w = 0;
            w_center = 0;
            w_total = 0;
            average = 0;
            //
            for (y_search = 0; y_search < D; y_search++) {
                for (x_search = 0; x_search < D; x_search++) {
                    area_tmp = area_search(Rect(x_search, y_search, d, d));
                    //计算欧式距离
                    d_eu = 0; cal = 0;//cal需要初始化么
                    for (int j = 0; j < d; j++) {
                        for (int i = 0; i < d; i++) {
                            if ((x_search + i == R + r) && (y_search + j == R + r)) continue;
                            cal = (int)(area_template.at<uchar>(j, i) - area_tmp.at<uchar>(j, i));
                            cal *= cal;
                            d_eu += cal * kernel.at<double>(j, i);
                        }
                    }
                    //
                    w = exp(-d_eu / (h * h));
                    w_center = w > w_center ? w : w_center;
                    w_total += w;
                    average += w * (int)area_tmp.at<uchar>(r, r);
                }
            }
            w_total += w_center;
            average += w_center * (int)area_search.at<uchar>(R + r, R + r);//area_template(r,r)可能也行
            if (w_total > 0)
                dst.at<uchar>(y_dst, x_dst) = (uchar)(average / w_total);
            else
                dst.at<uchar>(y_dst, x_dst) = area_template.at<uchar>(y_dst + r, x_dst + r);
        }
    }
    return 0;
}

//已成功；at法 速度慢；取坐标法 速度快；仅处理单通道图像
int NLM_withsigma_at(Mat& src_i, Mat& dst, double sigma, double h = 5, int r = 3, int R = 10) {
    int d = 2 * r + 1, D = 2 * R + 1;
    double sigma2 = sigma * sigma;
    Mat src;
    copyMakeBorder(src_i, src, R + r, R + r, R + r, R + r, BORDER_REFLECT);
    Mat kernel;
    //kernel = kernel.ones(2 * r + 1, 2 * r + 1, CV_64F) / ((2*r+1)*(2*r+1));
    kernel = getGaussianKernel(d, 1) * getGaussianKernel(d, 1).t();

    Mat area_search, area_template, area_tmp;
    int row_dst = src_i.rows, col_dst = src_i.cols, row_src = src.rows, col_src = src.cols;
    int x_dst, y_dst, x_src, y_src;
    int x_search, y_search, x_template, y_template;
    double w, w_center, w_total;
    double d_eu, cal;
    double average;
    float rate;
    for (y_dst = 0; y_dst < row_dst; y_dst++) {
        if (y_dst % 10 == 0) { rate = (float)y_dst / (float)row_dst * 100; cout << rate << "%" << endl; }
        for (x_dst = 0; x_dst < col_dst; x_dst++) {
            //area_template = src(Rect(x_dst + R, y_dst + R, d, d));//起始坐标待核实
            //area_search = src(Rect(x_dst, y_dst, D + d - 1, D + d - 1));
            x_template = x_dst + R;
            y_template = y_dst + R;

            //初始化待处理
            w = 0;
            w_center = 0;
            w_total = 0;
            average = 0;

            for (y_search = 0; y_search < D; y_search++) {
                for (x_search = 0; x_search < D; x_search++) {
                    //area_tmp = area_search(Rect(x_search, y_search, d, d));
                    //计算欧式距离
                    d_eu = 0; cal = 0;//cal需要初始化么
                    for (int j = 0; j < d; j++) {
                        for (int i = 0; i < d; i++) {
                            if ((x_search + i == R + r) && (y_search + j == R + r)) continue;

                            //cal = (int)(area_template.at<uchar>(j, i) - area_tmp.at<uchar>(j, i));
                            cal = (int)src.at<uchar>(y_template + j, x_template + i) - (int)src.at<uchar>(y_dst + y_search + j, x_dst + x_search + i);
                            cal *= cal;
                            d_eu += cal * kernel.at<double>(j, i);
                        }
                    }
                    //
                    d_eu = (d_eu - 2 * sigma2 > 0) ? d_eu : 0;
                    w = exp(-d_eu / (h * h));//除法h2待优化为乘法
                    w_center = w > w_center ? w : w_center;
                    w_total += w;
                    //average += w * (int)area_tmp.at<uchar>(r, r);
                    average += w * (int)src.at<uchar>(y_dst + y_search + r, x_dst + x_search + r);
                }
            }
            w_total += w_center;
            //average += w_center * (int)area_search.at<uchar>(R + r, R + r);//area_template(r,r)可能也行
            average += w_center * (int)src.at<uchar>(y_template + r, x_template + r);
            if (w_total > 0)
                dst.at<uchar>(y_dst, x_dst) = (uchar)(average / w_total);
            else
                //dst.at<uchar>(y_dst, x_dst) = area_template.at<uchar>(y_dst + r, x_dst + r);
                dst.at<uchar>(y_dst, x_dst) = src.at<uchar>(y_template + r, x_template + r);
        }
    }
    return 0;
}

//已成功；data法 速度快；取坐标法 速度快；仅处理单通道图像
int NLM_withsigma_data(Mat& src_i, Mat& dst, double sigma, double h = 5, int r = 3, int R = 10) {
    int d = 2 * r + 1, D = 2 * R + 1;
    double sigma2 = sigma * sigma, dh2 = 1 / (h * h);
    Mat src;
    copyMakeBorder(src_i, src, R + r, R + r, R + r, R + r, BORDER_REFLECT);
    Mat kernel;
    //kernel = kernel.ones(2 * r + 1, 2 * r + 1, CV_64F) / ((2*r+1)*(2*r+1));
    kernel = getGaussianKernel(d, 1) * getGaussianKernel(d, 1).t();

    //Mat area_search, area_template, area_tmp;

    //uchar* data_template = area_template.data, * data_tmp = area_tmp.data;
    uchar* data_src = src.data;
    uchar* data_kernel = kernel.data, * data_dst = dst.data;

    int row_dst = src_i.rows, col_dst = src_i.cols, row_src = src.rows, col_src = src.cols;
    int x_dst, y_dst, x_src, y_src;
    int x_search, y_search, x_template, y_template;
    double w, w_center, w_total;
    double d_eu, cal;
    double average;
    float rate;
    for (y_dst = 0; y_dst < row_dst; y_dst++) {
        if (y_dst % 10 == 0) { rate = (float)y_dst / (float)row_dst * 100; cout << rate << "%" << endl; }
        for (x_dst = 0; x_dst < col_dst; x_dst++) {
            //area_template = src(Rect(x_dst + R, y_dst + R, d, d));//起始坐标待核实
            //area_search = src(Rect(x_dst, y_dst, D + d - 1, D + d - 1));
            x_template = x_dst + R;
            y_template = y_dst + R;
            //初始化待处理
            w = 0;
            w_center = 0;
            w_total = 0;
            average = 0;
            //
            for (y_search = 0; y_search < D; y_search++) {
                for (x_search = 0; x_search < D; x_search++) {
                    //area_tmp = area_search(Rect(x_search, y_search, d, d));

                    //计算欧式距离
                    d_eu = 0; cal = 0;//cal需要初始化么
                    for (int j = 0; j < d; j++) {
                        for (int i = 0; i < d; i++) {
                            if ((x_search + i == R + r) && (y_search + j == R + r)) continue;
                            //cal = (int)(area_template.at<uchar>(j, i) - area_tmp.at<uchar>(j, i));
                            //cal = (int)src.at<uchar>(y_template + j, x_template + i) - (int)src.at<uchar>(y_dst+y_search+j, x_dst+x_search+i);
                            cal = (int)(*(data_src + (y_template + j) * src.step[0] + (x_template + i) * src.step[1])) - (int)(*(data_src + (y_dst + y_search + j) * src.step[0] + (x_dst + x_search + i) * src.step[1]));
                            cal *= cal;
                            //d_eu += cal * kernel.at<double>(j, i);
                            d_eu += cal * (*(double*)(data_kernel + j * kernel.step[0] + i * kernel.step[1]));
                        }
                    }
                    //
                    d_eu = (d_eu - 2 * sigma2 > 0) ? d_eu : 0;
                    w = exp(-d_eu * dh2);
                    w_center = w > w_center ? w : w_center;
                    w_total += w;
                    //average += w * (int)area_tmp.at<uchar>(r, r);
                    //average += w * (int)src.at<uchar>(y_dst+y_search+r, x_dst+x_search+r);
                    average += w * (int)(*(data_src + (y_dst + y_search + r) * src.step[0] + (x_dst + x_search + r) * src.step[1]));
                }
            }
            w_total += w_center;
            //average += w_center * (int)area_search.at<uchar>(R + r, R + r);//area_template(r,r)可能也行
            //average += w_center * (int)src.at<uchar>(y_template+r, x_template+r);
            average += w_center * (int)(*(data_src + (y_template + r) * src.step[0] + (x_template + r) * src.step[1]));
            if (w_total > 0)
                //dst.at<uchar>(y_dst, x_dst) = (uchar)(average / w_total);
                *(data_dst + y_dst * dst.step[0] + x_dst * dst.step[1]) = (uchar)(average / w_total);
            else
                //dst.at<uchar>(y_dst, x_dst) = area_template.at<uchar>(y_dst + r, x_dst + r);
                //dst.at<uchar>(y_dst, x_dst) = src.at<uchar>(y_template+r, x_template+r);
                *(data_dst + y_dst * dst.step[0] + x_dst * dst.step[1]) = *(data_src + (y_template + r) * src.step[0] + (x_template + r) * src.step[1]);
        }
    }
    return 0;
}

//已成功；at法 速度慢；截取法 速度慢；处理任意通道图像
int NLM_color(Mat& src_i, Mat& dst, double sigma, int h = 5, int r = 3, int R = 10) {
    int d = 2 * r + 1, D = 2 * R + 1;
    int channel = src_i.channels();
    double sigma2 = sigma * sigma;
    Mat src;
    copyMakeBorder(src_i, src, R + r, R + r, R + r, R + r, BORDER_REFLECT);
    Mat kernel;
    //kernel = kernel.ones(2 * r + 1, 2 * r + 1, CV_64F) / ((2*r+1)*(2*r+1));
    kernel = getGaussianKernel(d, 1) * getGaussianKernel(d, 1).t();

    Mat area_search, area_template, area_tmp;
    int row_dst = src_i.rows, col_dst = src_i.cols, row_src = src.rows, col_src = src.cols;
    int x_dst, y_dst, x_src, y_src, x_min, x_max, y_min, y_max;
    int x_search, y_search;
    double w, w_center, w_total;
    double d_eu, cal;
    double average[4] = { 0 };
    float rate;
    for (y_dst = 0; y_dst < row_dst; y_dst++) {
        if (y_dst % 10 == 0) { rate = (float)y_dst / (float)row_dst * 100; cout << rate << "%" << endl; }
        for (x_dst = 0; x_dst < col_dst; x_dst++) {
            area_template = src(Rect(x_dst + R, y_dst + R, d, d));//起始坐标待核实
            area_search = src(Rect(x_dst, y_dst, D + d - 1, D + d - 1));
            //uchar m = area_template.at<uchar>(r, r), n = area_search.at<uchar>(R + r, R + r);
            //初始化待处理
            w = 0;
            w_center = 0;
            w_total = 0;
            for (int i = 0; i < 4; i++) { average[i] = 0; }
            //
            for (y_search = 0; y_search < D; y_search++) {
                for (x_search = 0; x_search < D; x_search++) {
                    area_tmp = area_search(Rect(x_search, y_search, d, d));
                    //计算欧式距离
                    d_eu = 0; cal = 0;//cal需要初始化么
                    for (int j = 0; j < d; j++) {
                        for (int i = 0; i < d; i++) {
                            if ((x_search + i == R + r) && (y_search + j == R + r)) continue;
                            for (int p = 0; p < channel; p++) {
                                cal = (int)(area_template.at<uchar>(j, i * channel + p) - area_tmp.at<uchar>(j, i * channel + p));
                                cal *= cal;
                                d_eu += cal * kernel.at<double>(j, i);
                            }
                        }
                    }
                    d_eu /= 3;
                    d_eu = (d_eu - 2 * sigma2 > 0) ? d_eu : 0;
                    w = exp(-d_eu / (h * h));
                    w_center = w > w_center ? w : w_center;
                    w_total += w;
                    for (int i = 0; i < channel; i++) {
                        average[i] += w * (int)area_tmp.at<uchar>(r, r * channel + i);
                    }
                }
            }
            w_total += w_center;
            for (int i = 0; i < channel; i++) { average[i] += w_center * (int)area_search.at<uchar>(R + r, (R + r) * channel + i); }//area_template(r,r)可能也行
            if (w_total > 0)
                for (int i = 0; i < channel; i++) {
                    dst.at<uchar>(y_dst, x_dst * channel + i) = (uchar)(average[i] / w_total);
                }
            else
                for (int i = 0; i < channel; i++) {
                    dst.at<uchar>(y_dst, x_dst * channel + i) = area_template.at<uchar>(y_dst + r, (x_dst + r) * channel + i);
                }
        }
    }
    return 0;
}

//最原版 不成功；初步探索算法
int none_local_means(Mat& src, Mat& dst, int h = 3, int t = 3, int s = 10) {
    //Mat src = copyMakeBorder(src_i,src,)
    Mat kernel;
    Mat area_template, area_search;
    //kernel = kernel.ones(2*t+1, 2*t+1, CV_64F) / ((2*t+1)^2);
    kernel = getGaussianKernel(2 * t + 1, 1) * getGaussianKernel(2 * t + 1, 1).t();
    double to = 0;
    for (int i = 0; i < kernel.rows; i++) {
        for (int j = 0; j < kernel.cols; j++) {
            to += kernel.at<double>(i, j);
        }
    }
    cout << "kerneltotal=" << to << endl;
    //Mat tem;
    int m = src.rows, n = src.cols;
    int x, y, x_center, y_center, x_max, x_min, y_max, y_min;
    int i, j, p, q;
    double w, w_total, w_center;
    double d_eu;
    double tem;
    double average;
    float rate;
    cout << "dst size=" << dst.size << endl;
    for (y = 0; y < m - 2 * t - 1; y++) {
        if (y % 50 == 0) {
            rate = (float)y / (float)(m - 2 * t - 1) * 100;
            cout << rate << "%" << endl;
        }
        for (x = 0; x < n - 2 * t - 1; x++) {
            //cout << "x=" << x << endl;
            x_center = x + t;
            y_center = y + t;
            //if ((x_center + t) < m && (y_center))
            area_template = src(Rect(x_center - t, y_center - t, 2 * t + 1, 2 * t + 1));
            w_center = 0;
            average = 0;
            w_total = 0;
            x_min = max(x_center - s, t);//边界处理有bug
            x_max = min(x_center + s, n - t - 1);
            y_min = max(y_center - s, t);
            y_max = min(y_center + s, m - t - 1);
            //cout << "x_max=" << x_max << endl;
            for (j = y_min; j < y_max; j++) {
                for (i = x_min; i < x_max; i++) {
                    //if (x_max==n-t-1)
                        //cout << "i=" << i << endl;
                    if (i == x_center && j == y_center) continue;
                    //debug();
                    area_search = src(Rect(i - t, j - t, 2 * t + 1, 2 * t + 1));
                    w = 0;
                    d_eu = 0;
                    //tem = area_search - area_template;//会不会有截断bug？
                    //multiply(tem, tem, tem);
                    //multiply(kernel, tem, tem);
                    for (p = 0; p < 2 * t + 1; p++) {
                        for (q = 0; q < 2 * t + 1; q++) {
                            tem = (int)area_search.at<uchar>(q, p) - (int)area_template.at<uchar>(q, p);
                            tem *= tem;
                            tem *= (double)kernel.at<double>(q, p);
                            d_eu += tem;
                        }
                    }
                    w = exp(-d_eu / (h ^ 2));
                    if (w > w_center) w_center = w;
                    w_total += w;
                    average += w * src.at<uchar>(j, i);
                }
            }
            w_total += w_center;
            average += w_center * (int)src.at<uchar>(y_center, x_center);
            if (w_total > 0)
                dst.at<uchar>(y, x) = (uchar)(average / w_total);
            else
                dst.at<uchar>(y, x) = src.at<uchar>(y, x);
        }
    }
    cout << "over" << endl;
    return 0;
}