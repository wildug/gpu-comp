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

#include <cublasLt.h>
#include <cuda_runtime_api.h>

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

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %s\n", cublasLtGetStatusString(status));
        throw std::logic_error("cuBLAS API failed");
    }
}

struct AbsValue {
    __host__ __device__
    float operator()(const int32_t& x) const {
        return abs(x);
    }
};

float absMaxWithThrustDevice(int32_t* d_input, int n) {
    thrust::device_ptr<int32_t> dev_ptr(d_input);

    return thrust::transform_reduce(
        dev_ptr, dev_ptr + n,
        AbsValue(),              // transform: fabs(x)
        0.0f,                    // init
        thrust::maximum<int32_t>() // reduce: max
    );
}

__global__ void normalizeAndRoundtoInt8(int32_t* res32, int8_t* res8, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int32_t a = res32[idx];
        float afl = static_cast<float>(a);

        a = __float2int_rn(afl/ scalar);
        res8[idx] = static_cast<int8_t>(a);
    }
}


class Matrix{
public:
    uint32_t rows;
    uint32_t cols;
    float w_delta;
    int8_t* data;

    Matrix(uint32_t r, uint32_t c, float w_delta, int8_t* d) : rows(r), cols(c), w_delta(w_delta), data(d) {}

    __host__ static Matrix deserialize(std::ifstream& file){
        uint32_t rows, cols;
        float grid_spacing;

        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        file.read(reinterpret_cast<char*>(&grid_spacing), sizeof(float));
        
        int8_t*  int_data;
        cudaMallocHost(&int_data, rows*cols*sizeof(int8_t)) ;
        printf("rows: %d, cols: %d\n",rows, cols);
        file.read(reinterpret_cast<char*>(int_data), rows*cols*sizeof(int8_t));

        

        if ((cols * rows) % 2 == 1){
            file.seekg(1, std::ios::cur);
            printf("Odd number of Matrix elements: skipped padding.");
        }

        return Matrix(rows, cols, grid_spacing, int_data);
    }
    
        float mult(cublasLtHandle_t ltHandle, int32_t* d_result32, int8_t* result, int8_t* vector, float v_delta, void *workspace, cublasLtMatmulAlgo_t algo);
};

float Matrix::mult(cublasLtHandle_t ltHandle, int32_t* d_result32, int8_t* result, int8_t* vector, float v_delta, void *workspace, cublasLtMatmulAlgo_t algo){
    // returns float value 

    int32_t alpha = 1;  // Scale of vector and matrix quantization
    int32_t beta = 0.0f;   // Scalar multiplier for the initial value of y (should be 0 if we're just doing the product)
    int rows = this->rows;         // Leading dimension of matrix A
    int cols = this->cols;
    int num_elems = this->cols*this->rows;



    checkCUDAError("Before Sgemv");

    cublasLtMatrixLayout_t Adesc, Bdesc, Ddesc;
    cublasLtMatmulDesc_t matmulDesc;

    int m = rows, n = 1, k = cols;
    int lda = rows, ldb = cols, ldc = rows;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, ldb);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32I, m, n, ldc);
    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
 

    checkCublasStatus(
    cublasLtMatmul(
            ltHandle,
            matmulDesc,
            &alpha,
            this->data, Adesc,
            vector, Bdesc,
            &beta,
            d_result32, Ddesc,
            d_result32, Ddesc,
            &algo,
            workspace, 1<<20,
            0
        ));
    
    checkCUDAError("after Sgemv");
    float abs_max = absMaxWithThrustDevice(d_result32, this->rows);

    v_delta = abs_max / 127;

    int blocks = (rows+ MAX_THREADS - 1) / MAX_THREADS;

    normalizeAndRoundtoInt8<<< blocks,MAX_THREADS>>>
    (d_result32, result, v_delta, this->rows);
    checkCUDAError("after normalizeAndRound");


    return v_delta;
}
int main(int argc, char* argv[]) {

    std::string filename = "/home/wildug/RSP/myKernel/raw-matrices.bin";

    // std::string filename = "/home/ludwigal/readMat/compressed_matrices.bin";
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file" << std::endl;
        return 1;
    }


    // Initialize cuBLAS handle
    float ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // cuBLASLt handle
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    uint32_t num_matrices, len_v;
    int8_t* h_vec;

    file.read(reinterpret_cast<char*>(&num_matrices), sizeof(num_matrices));
    file.read(reinterpret_cast<char*>(&len_v), sizeof(len_v));

    h_vec = new int8_t[len_v];
    std::cout << "Number of matrices: " << num_matrices << std::endl;
    std::cout << "len_v: " << len_v << std::endl;
    file.read(reinterpret_cast<char*>(h_vec), len_v*sizeof(uint8_t));

    int8_t* vec;


    printf("end\n");
    cudaMalloc(&vec, sizeof(int8_t)*len_v);
    cudaMemcpy(vec, h_vec, sizeof(int8_t)*len_v,cudaMemcpyHostToDevice);

    checkCUDAError("after Reading");

    std::vector<Matrix> matrices;
    for (int k = 0; k<num_matrices; k++){
        printf("%d\n",k);
        Matrix matrix = Matrix::deserialize(file);
        matrices.push_back(std::move(matrix));
    }

    file.close();
    cudaEventRecord(start);

    for (int k = 0; k<num_matrices; k++){
        Matrix& matrix = matrices[k];
        int8_t* d_data;
        int num_elems = matrix.rows*matrix.cols;

        checkCUDAError("before Malloc");
        cudaMalloc(&d_data, sizeof(int8_t)*num_elems);
        checkCUDAError("after Malloc");
        cudaMemcpy(d_data, matrix.data,sizeof(int8_t)*num_elems, cudaMemcpyHostToDevice);
        checkCUDAError("after Memcpy");


        matrix.data = d_data;
    }


    checkCUDAError("after loop");
    int max_rows = 0;
    for (const auto& matrix : matrices) {
        if (matrix.rows > max_rows)
            max_rows = matrix.rows;
    }

    printf("max_rows: %d\n", max_rows);

    int32_t* d_result32;
    int8_t* d_result8;
    int8_t* blob;
    cudaMalloc(&d_result32, sizeof(int32_t)*max_rows);
    cudaMalloc(&d_result8, sizeof(int8_t)*max_rows);

    checkCUDAError("after allocating d_result");
    int rows;
    int v_delta = 1; // scaling factor of v starts with 1

    // cuBLASlt
    int m = 4096, n = 1, k = 4096;
    int lda = 4096, ldb = 4096, ldc = 4096;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatmulDesc_t matmulDesc;

    // A: m x k, B: k x n, C: m x n
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc);
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    // Heuristic algorithm search
    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    void *workspace = nullptr;
    size_t workspaceSize = 1 << 22; // arbitrary

    cudaMalloc(&workspace, workspaceSize);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;


    cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
    algo = heuristicResult.algo;

    for (int k = 0; k<num_matrices; k++){
        Matrix matrix = matrices[k];
        rows = matrix.rows;
        v_delta = matrix.mult(ltHandle, d_result32, d_result8, vec, v_delta, workspace, algo);
        checkCUDAError("after multiplying matrix");
        blob = vec;
        vec = d_result8;
        d_result8 = blob;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f ms\n", ms);
    checkCUDAError("after loop");
    
    cudaMemcpy(h_vec, vec, sizeof(int8_t)*rows, cudaMemcpyDeviceToHost);
    cudaFree(vec);
    for (int k = 0; k<num_matrices; k++){
        Matrix matrix = matrices[k];
        cudaFree(matrix.data);
    }




    // Output result
    printf("[");
    for (int i=0; i<rows; i++){
        // printf("Result at index %d: %d\n", i, h_result[i]);
        printf("%d,",  h_vec[i]);
    }
    printf("]\n");



    cublasLtDestroy(ltHandle);

    cudaDeviceReset();
    checkCUDAError("End of program.");

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