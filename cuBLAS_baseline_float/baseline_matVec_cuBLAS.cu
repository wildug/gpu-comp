#include <cassert>
#include <cstdint>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <fstream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cmath>

#define EPSILON 1e-5 
#define MAX_BLOCKS 256
#define MAX_THREADS 256

#define VERBOSE

bool validateResults(float* hostMat, float* hostVec, float* hostResVec, int w, int h);
void checkCUDAError(const char* msg);

struct AbsValue {
    __host__ __device__
    float operator()(const float& x) const {
        return fabsf(x);
    }
};

uint32_t hash_int8_array(int8_t* arr, int size)
{
    uint32_t hash = 0;

    for (size_t i = 0; i < size; i++)
    {
        hash = (hash >> 27) | (hash << 5); // Rotate left by 5 bits
        hash = (hash ^ *reinterpret_cast<const uint8_t *>(&arr[i])) * 0x27220A95;
    }

    return hash;
}

float absMaxWithThrustDevice(float* d_input, int n) {
    thrust::device_ptr<float> dev_ptr(d_input);

    return thrust::transform_reduce(
        dev_ptr, dev_ptr + n,
        AbsValue(),              // transform: fabs(x)
        0.0f,                    // init
        thrust::maximum<float>() // reduce: max
    );
}

__global__ void normalizeAndRound(float* vec, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vec[idx] = roundf(vec[idx]/ scalar);
    }
}


class Matrix{
public:
    uint32_t rows;
    uint32_t cols;
    float w_delta;
    float* data;
    float* d_data;

    Matrix(uint32_t r, uint32_t c, float w_delta, float* d) : rows(r), cols(c), w_delta(w_delta), data(d) {}

    __host__ static Matrix deserialize(std::ifstream& file){
        uint32_t rows, cols;
        float grid_spacing;

        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        file.read(reinterpret_cast<char*>(&grid_spacing), sizeof(float));
        
        int8_t*  int_data = new int8_t[rows*cols];
        float* data; 
        printf("rows: %d, cols: %d\n",rows, cols);
        cudaHostAlloc(&data, rows*cols*sizeof(float), cudaHostAllocDefault);
        file.read(reinterpret_cast<char*>(int_data), rows*cols*sizeof(int8_t));

        
        // Casting to float here, watchout!
        for (int i = 0; i < rows*cols; ++i) {
                data[i] = static_cast<float>(int_data[i]);
        }

        if ((cols * rows) % 2 == 1){
            file.seekg(1, std::ios::cur);
            printf("Odd number of Matrix elements: skipped padding.");
        }

        delete [] int_data;

        return Matrix(rows, cols, grid_spacing, data);
    }
    
    float mult(cublasHandle_t handle, float* result, float* vector, float v_delta);
};

float Matrix::mult(cublasHandle_t handle, float* result, float* vector, float v_delta){
    // returns float value 

    float alpha = this->w_delta * v_delta;  // Scale of vector and matrix quantization
    float beta = 0.0f;   // Scalar multiplier for the initial value of y (should be 0 if we're just doing the product)
    int rows = this->rows;         // Leading dimension of matrix A
    int incx = 1;        // Increment for vector x
    int incy = 1;        // Increment for vector y
    int num_elems = this->cols*this->rows;



    cublasStatus_t stat;
    checkCUDAError("Before Sgemv");

    stat = cublasSgemv(handle, CUBLAS_OP_T, rows, this->cols, &alpha, this->d_data, rows, vector, incx, &beta, result, incy);
    
    checkCUDAError("after Sgemv");
    float abs_max = absMaxWithThrustDevice(result, this->rows);

    v_delta = abs_max / 127;

    int blocks = (rows+ MAX_THREADS - 1) / MAX_THREADS;

    normalizeAndRound<<< blocks,MAX_THREADS>>>    (result, v_delta, rows);
    checkCUDAError("after normalizeAndRound");


    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR CUBLAS_STATUS_SUCCESS)");
    }

    return v_delta;
}
int main(int argc, char* argv[]) {

    std::string filename = "/home/wildug/RSP/myKernel/raw-matrices_4096.bin";
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file" << std::endl;
        return 1;
    }


    // Initialize cuBLAS handle
    float ms1 = 0;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    // time using on
    float ms2 = 0;
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);

    cublasHandle_t handle;
    cublasCreate(&handle);

    uint32_t num_matrices, len_v, result_hash;
    int8_t* int_vec;

    file.read(reinterpret_cast<char*>(&num_matrices), sizeof(num_matrices));
    file.read(reinterpret_cast<char*>(&result_hash), sizeof(result_hash));
    file.read(reinterpret_cast<char*>(&len_v), sizeof(len_v));

    int_vec = new int8_t[len_v];
    std::cout << "Number of matrices: " << num_matrices << std::endl;
    std::cout << "len_v: " << len_v << std::endl;
    file.read(reinterpret_cast<char*>(int_vec), len_v*sizeof(uint8_t));

    float* h_vec = new float[len_v];
    float* vec;
    float* v0;


    for (int i = 0; i < len_v; ++i) {
            h_vec[i] = static_cast<float>(int_vec[i]);
            // printf("%f, ",h_vec[i]);
    }
    printf("end\n");
    cudaMalloc(&vec, sizeof(float)*len_v);
    cudaMalloc(&v0, sizeof(float)*len_v);
    cudaMemcpy(v0, h_vec, sizeof(float)*len_v,cudaMemcpyHostToDevice);

    checkCUDAError("after Reading");

    std::vector<Matrix> matrices;
    for (int k = 0; k<num_matrices; k++){
        printf("%d\n",k);
        Matrix matrix = Matrix::deserialize(file);
        matrices.push_back(std::move(matrix));
    }
    file.close();

    for (int l=0; l< 20; l++){ // outer loop for benchmarking
        cudaEventRecord(start1);
        // copy initial array to vec
        cudaMemcpy(vec, v0, sizeof(float)*len_v,cudaMemcpyDeviceToDevice);

        for (int k = 0; k<num_matrices; k++){
            Matrix& matrix = matrices[k];
            float* d_data;
            int num_elems = matrix.rows*matrix.cols;

            checkCUDAError("before Malloc");
            cudaMalloc((void**)&d_data, sizeof(float)*num_elems);
            checkCUDAError("after Malloc");
            cudaMemcpy(d_data, matrix.data,sizeof(float)*num_elems, cudaMemcpyHostToDevice);
            checkCUDAError("after Memcpy");


            matrix.d_data = d_data;
        }


        checkCUDAError("after loop");
        int max_rows = 0;
        for (const auto& matrix : matrices) {
            if (matrix.rows > max_rows)
                max_rows = matrix.rows;
        }


        float* d_result;
        float* blob;
        cudaMalloc(&d_result, sizeof(float)*max_rows);

        checkCUDAError("after allocating d_result");
        int rows;

        cudaEventRecord(start2);
        int v_delta = 1; // scaling factor of v starts with 1

        for (int k = 0; k<num_matrices; k++){
            Matrix matrix = matrices[k];
            rows = matrix.rows;
            v_delta = matrix.mult(handle, d_result, vec, v_delta);

            checkCUDAError("after multiplying matrix");
            // to swap variables you need a third guy 'blob'
            blob = vec;
            vec = d_result;
            d_result = blob;
        }
        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);
        cudaMemcpy(h_vec, vec, sizeof(float)*rows, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);


        cudaEventElapsedTime(&ms1, start1, stop1);
        cudaEventElapsedTime(&ms2, start2, stop2);
        
        for (int k = 0; k<num_matrices; k++){
            Matrix matrix = matrices[k];
            cudaFree(matrix.d_data);
        }

        printf("Time with memcpy:    %f ms\n", ms1);
        printf("Time without memcpy: %f ms\n", ms2);
    }

    cudaFree(vec);



    // Output result
    std::cout << "Result vector y: ";
    for (int i = 0; i < len_v; i++) {
        std::cout << h_vec[i] << ", ";
    }
    std::cout << std::endl;



    cublasDestroy(handle);

    return 0;
}

bool validateResults(float* hostMat, float* hostVec, float* gpuResult, int w, int h) {

    float* result = new float[h];

    for (int i = 0; i < h; ++i) {
        result[i] = 0.0f;
        for (int j = 0; j < w; ++j) {
            result[i] += hostMat[i * w + j] * hostVec[j];
        }
    }
    for (int i = 0; i < h; ++i) {
        if (fabs(result[i] - gpuResult[i]) > EPSILON) {
            printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, result[i], gpuResult[i]);
            return false;
        }
    }
    delete [] result;
    return true;
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}