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

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define EPSILON 1e-5 
#define MAX_BLOCKS 256
#ifndef MAX_THREADS
#define MAX_THREADS 128
#endif

#define VERBOSE

bool validateResults(float* hostMat, float* hostVec, float* hostResVec, int w, int h);
void checkCUDAError(const char* msg);

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %s\n", cublasLtGetStatusString(status));
        throw std::logic_error("cuBLAS API failed");
    }
}

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
        thrust::maximum<float>() // reduce: max
    );
}

__global__ void normalizeAndRoundtoInt8(int32_t* res32, int8_t* res8, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int32_t a = res32[idx];
        float afl = static_cast<float>(a);

        a = __float2int_rn(afl/ scalar);
        a = max(-128, min(127, a));
        res8[idx] = static_cast<int8_t>(a);
    }
}


// CHAT CODE

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif





__global__ void vecMat_int8_warpRows(
    int32_t* __restrict__ dst,
    const int8_t* __restrict__ mat, // row-major [h x w]
    const int8_t* __restrict__ v,
    int w, int h)
{
    // Warp bookkeeping
    const int tid   = threadIdx.x;
    const int warp  = tid / WARP_SIZE;
    const int lane  = tid % WARP_SIZE;

    // One warp == one row
    const int warpsPerBlock = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warpsPerBlock + warp;
    if (row >= h) return;

    // ---- Load vector into shared (int32-packed) once per block ----
    extern __shared__ __align__(16) int32_t sV32[];       // size >= w/4
    int4* sV128 = reinterpret_cast<int4*>(sV32);

    // Cooperatively load with 128-bit transactions
    const int nChunks128 = w / 16;          // 16 bytes per int4
    for (int i = tid; i < nChunks128; i += blockDim.x) {
        const int4 vv = *(reinterpret_cast<const int4*>(v) + i);
        sV128[i] = vv; // 16B store to shared
    }
    __syncthreads();

    // Reinterpret row and shared vector as int32 streams
    const int4* row4 = reinterpret_cast<const int4*>(mat + size_t(row) * w);
    const int4* v4_shared = sV128;


    // Each lane walks over int4 chunks with stride = warpSize
    int acc = 0;
    // # pragma unroll 4
    for (int j = lane; j < nChunks128; j += WARP_SIZE) {
        // load 16B (4x int32 packed int8) from row and vector
        const int4 a4 = __ldg(&row4[j]);
        const int4 b4 = v4_shared[j];

        // 4 dp4a per 128-bit chunk
        acc = __dp4a(a4.x, b4.x, acc);
        acc = __dp4a(a4.y, b4.y, acc);
        acc = __dp4a(a4.z, b4.z, acc);
        acc = __dp4a(a4.w, b4.w, acc);
    }

    // Warp reduction (sum partials across lanes)
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }


    // Lane 0 writes the row result
    if (lane == 0) dst[row] = acc;
}

__global__ void vecMat_int8_2warpRows(
    int32_t* __restrict__ dst,
    const int8_t* __restrict__ mat, // row-major [h x w]
    const int8_t* __restrict__ v,
    int w, int h)
{
    // Warp bookkeeping
    const int tidx   = threadIdx.x;
    const int tidy   = threadIdx.y;
    const int warp  = tidx / WARP_SIZE;
    const int lane  = tidx % WARP_SIZE;
    //
    // into how many peaces do we cut each row
    const int cuts = blockDim.y;

    // One block == one row
    const int warpsPerBlock = (blockDim.x * blockDim.y) / WARP_SIZE;
    const int row = (blockIdx.x * blockDim.x + warp);
    if (row >= h) return;

    // ---- Load vector into shared (int32-packed) once per block ----
    extern __shared__ __align__(16) int32_t sV32[];       // size >= w/4
    int4* sV128 = reinterpret_cast<int4*>(sV32);

    // Cooperatively load with 128-bit transactions
    const int nChunks128 = w / (16*cuts);          // 16 bytes per int4
    for (int i = tidx+ tidy * nChunks128; i < (tidy+1) * nChunks128; i += blockDim.x) {
        const int4 vv = *(reinterpret_cast<const int4*>(v) + i);
        sV128[i] = vv; // 16B store to shared
    }
    __syncthreads();

    // Reinterpret row and shared vector as int32 streams
    const int4* row4 = reinterpret_cast<const int4*>(mat + size_t(row) * w);
    const int4* v4_shared = sV128;


    // Each lane walks over int4 chunks with stride = warpSize
    int acc = 0;
    for (int j = lane; j < nChunks128; j += WARP_SIZE) {
        // load 16B (4x int32 packed int8) from row and vector
        const int4 a4 = __ldg(&row4[j]);
        const int4 b4 = v4_shared[j];

        // 4 dp4a per 128-bit chunk
        acc = __dp4a(a4.x, b4.x, acc);
        acc = __dp4a(a4.y, b4.y, acc);
        acc = __dp4a(a4.z, b4.z, acc);
        acc = __dp4a(a4.w, b4.w, acc);
    }

    // Warp reduction (sum partials across lanes)
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }


    // Each warp leader writes to shared memory
    __shared__ int warpSums[32];  // enough for up to 1024 threads
    if (lane == 0) warpSums[warp] = acc;
    __syncthreads();

    // Let warp 0 reduce the warpSums
    if (warp == 0) {
        int blockAcc = (tidx < (blockDim.x >> 5)) ? warpSums[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            blockAcc += __shfl_down_sync(0xffffffff, blockAcc, offset);
        }
        if (lane == 0) dst[row] = blockAcc;
    }
}

__global__ void vecMat_int8_blockRow(
    int32_t* __restrict__ dst,
    const int8_t* __restrict__ mat, // [h x w], row-major
    const int8_t* __restrict__ v,
    int w, int h)
{
    extern __shared__ int32_t sV32[];  // shared vector copy (ceil(w/4))
    int4* sV128 = reinterpret_cast<int4*>(sV32);

    const int row = blockIdx.x;        // one row per block
    if (row >= h) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid>> 5;

    // ---- Copy vector v into shared memory cooperatively ----
    const int nChunks128 = w / 16;
    for (int i = tid; i < nChunks128; i += blockDim.x) {
        sV128[i] = reinterpret_cast<const int4*>(v)[i];
    }
    __syncthreads();

    // ---- Compute local partial sum ----
    int acc = 0;
    const int4* row4 = reinterpret_cast<const int4*>(mat + size_t(row) * w);

    // stride through int4 chunks
    for (int j = tid; j < nChunks128; j += blockDim.x) {
        const int4 a4 = row4[j];
        const int4 b4 = sV128[j];
        acc = __dp4a(a4.x, b4.x, acc);
        acc = __dp4a(a4.y, b4.y, acc);
        acc = __dp4a(a4.z, b4.z, acc);
        acc = __dp4a(a4.w, b4.w, acc);
    }

    // Handle tail (when w not divisible by 16)
    const int tailStart = nChunks128 * 16;
    for (int k = tailStart + tid; k < w; k += blockDim.x) {
        acc += int(mat[row * w + k]) * int(v[k]);
    }

    // ---- Block-wide reduction ----
    // First do warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Each warp leader writes to shared memory
    __shared__ int warpSums[32];  // enough for up to 1024 threads
    if (lane == 0) warpSums[warp] = acc;
    __syncthreads();

    // Let warp 0 reduce the warpSums
    if (warp == 0) {
        // if tid smaller than blockDim.x / 32 access warpSums[lane]
        int blockAcc = (tid < (blockDim.x >> 5)) ? warpSums[lane] : 0;
        // 16 is warpsize / 2
        for (int offset = 16; offset > 0; offset >>= 1) {
            blockAcc += __shfl_down_sync(0xffffffff, blockAcc, offset);
        }
        if (lane == 0) dst[row] = blockAcc;
    }
}

__global__ void vecMat_int8_blockRow2(
    int32_t* __restrict__ dst,
    const int8_t* __restrict__ mat, // [h x w], row-major
    const int8_t* __restrict__ v,
    int w, int h)
{
    extern __shared__ int32_t sV32[];  // shared vector copy (ceil(w/4))
    int4* sV128 = reinterpret_cast<int4*>(sV32);

    const int cuts = blockDim.y;        // how many cuts along row

    const int row = blockIdx.x;        // one row per block
    const int cut = blockIdx.y;        // which cut along row
    if (row >= h) return;

    const int tidx = threadIdx.x;
    const int tidy  = threadIdx.y;
    const int lane = tidx & 31;
    const int warp = (tidx >> 5) + cut * blockDim.x;

    // ---- Copy vector v into shared memory cooperatively ----
    const int nChunks128 = (w / (16 * cuts)) * (cut + 1);
    const int start = (w / (16 * cuts)) * cut + tidx;
    for (int i = start; i < nChunks128; i += blockDim.x) {
        sV128[i] = reinterpret_cast<const int4*>(v)[i];
    }
    __syncthreads();

    // ---- Compute local partial sum ----
    int acc = 0;
    const int4* row4 = reinterpret_cast<const int4*>(mat + size_t(row) * w);

    // stride through int4 chunks
    for (int j = start; j < nChunks128; j += blockDim.x) {
        const int4 a4 = row4[j];
        const int4 b4 = sV128[j];
        acc = __dp4a(a4.x, b4.x, acc);
        acc = __dp4a(a4.y, b4.y, acc);
        acc = __dp4a(a4.z, b4.z, acc);
        acc = __dp4a(a4.w, b4.w, acc);
    }


    // ---- Block-wide reduction ----
    // First do warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Each warp leader writes to shared memory
    __shared__ int warpSums[32];  // enough for up to 1024 threads
    if (lane == 0) warpSums[warp] = acc;
    __syncthreads();

    // Let warp 0 reduce the warpSums
    if (warp == 0) {
        // if tid smaller than blockDim.x / 32 access warpSums[lane]
        int blockAcc = (tidx < (blockDim.x >> 5)) ? warpSums[lane] : 0;
        // 16 is warpsize / 2
        for (int offset = 16; offset > 0; offset >>= 1) {
            blockAcc += __shfl_down_sync(0xffffffff, blockAcc, offset);
        }
        if (lane == 0) dst[row] = blockAcc;
    }
}

class Matrix{
public:
    uint32_t rows;
    uint32_t cols;
    float w_delta;
    int8_t* data;
    int8_t* d_data;

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
    
        float mult(int32_t* d_result32, int8_t* result, int8_t* vector, float v_delta);
};

float Matrix::mult(int32_t* d_result32, int8_t* result, int8_t* vector, float v_delta){
    // returns float value 

    int rows = this->rows;         // Leading dimension of matrix A
    int cols = this->cols;



    checkCUDAError("Before Sgemv");
    int blocks = (rows+ MAX_THREADS - 1) / MAX_THREADS;

    // int warpsPerBlock = 4;
    // dim3 grid( (rows + warpsPerBlock - 1) / warpsPerBlock );
    // dim3 block(warpsPerBlock * 32);
    // vecMat_int8_warpRows<<<grid,block, cols*sizeof(int8_t)>>>(d_result32, this->d_data, vector, cols, rows);

    dim3 blockGrid(rows);
    dim3 threadBlock(MAX_THREADS);
    vecMat_int8_blockRow<<<blockGrid, threadBlock, cols*sizeof(int8_t)>>>(d_result32, this->d_data, vector, cols, rows);


    
    checkCUDAError("after Sgemv");
    float abs_max = absMaxWithThrustDevice(d_result32, this->rows);

    // printf("absmax: %f, ", abs_max);
    v_delta = abs_max / 127;


    normalizeAndRoundtoInt8<<< blocks,MAX_THREADS>>>
    (d_result32, result, v_delta, this->rows);
    checkCUDAError("after normalizeAndRound");


    return v_delta;
}
int main(int argc, char* argv[]) {

    // std::string filename = "/home/wildug/RSP/myKernel/raw-matrices_4096.bin";
    std::vector<std::string> filepaths = {
        "/home/wildug/RSP/myKernel/raw-matrices.bin",
        "/mnt/lustre/work/bamler/bdz937/RSP/matrices/raw-matrices_4096.bin",
        "/home/ludwigal/matrix_binfiles/raw-matrices_4096.bin",
    };


    //std::string filename = "/mnt/lustre/work/bamler/bdz937/RSP/matrices/raw-matrices_4096.bin";
    //std::string filename = "/home/ludwigal/cuBLAS_baseline/raw-matrices.bin";
    bool opened = false;
    std::ifstream file;
    for (const auto& path : filepaths) {
        file.open(path);
        if (file.is_open()) {
            opened = true;
            break;
        }
    }

    if (!opened) {
        std::cerr << "Error: Could not open file from any of the given paths." << std::endl;
        return 1;
    }



    // for timing
    // time including memcpy
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

    // Initialize cuBLAS handle
    // cuBLASLt handle

    uint32_t num_matrices, len_v, result_hash;
    int8_t* h_vec;

    file.read(reinterpret_cast<char*>(&num_matrices), sizeof(num_matrices));
    file.read(reinterpret_cast<char*>(&result_hash), sizeof(result_hash));
    file.read(reinterpret_cast<char*>(&len_v), sizeof(len_v));

    h_vec = new int8_t[len_v];
    std::cout << "Number of matrices: " << num_matrices << std::endl;
    std::cout << "len_v: " << len_v << std::endl;
    file.read(reinterpret_cast<char*>(h_vec), len_v*sizeof(uint8_t));

    int8_t* vec;
    int8_t* v0;


    printf("end\n");
    cudaMalloc(&v0, sizeof(int8_t)*len_v);
    cudaMalloc(&vec, sizeof(int8_t)*len_v);
    cudaMemcpy(v0, h_vec, sizeof(int8_t)*len_v,cudaMemcpyHostToDevice);

    checkCUDAError("after Reading");

    std::vector<Matrix> matrices;
    for (int k = 0; k<num_matrices; k++){
        printf("%d\n",k);
        Matrix matrix = Matrix::deserialize(file);
        matrices.push_back(std::move(matrix));
    }

    file.close();



    int max_rows = 0;
    for (const auto& matrix : matrices) {
        if (matrix.rows > max_rows)
            max_rows = matrix.rows;
    }


    for (int l=0; l< 20; l++){ // outer loop for benchmarking
        cudaEventRecord(start1);

        for (int k = 0; k<num_matrices; k++){
            Matrix& matrix = matrices[k];
            int8_t* d_data;
            int num_elems = matrix.rows*matrix.cols;

            checkCUDAError("before Malloc");
            cudaMalloc(&d_data, sizeof(int8_t)*num_elems);
            checkCUDAError("after Malloc");
            cudaMemcpy(d_data, matrix.data,sizeof(int8_t)*num_elems, cudaMemcpyHostToDevice);
            checkCUDAError("after Memcpy");


            matrix.d_data = d_data;
        }

        cudaMemcpy(vec, v0, sizeof(int8_t)*len_v, cudaMemcpyDeviceToDevice);

        checkCUDAError("after loop");


        int32_t* d_result32;
        int8_t* d_result8;
        int8_t* blob;
        cudaMalloc(&d_result32, sizeof(int32_t)*max_rows);
        cudaMalloc(&d_result8, sizeof(int8_t)*max_rows);

        checkCUDAError("after allocating d_result");
        int rows;
        cudaEventRecord(start2);
        int v_delta = 1; // scaling factor of v starts with 1


        for (int k = 0; k<num_matrices; k++){
            Matrix matrix = matrices[k];
            rows = matrix.rows;
            v_delta = matrix.mult(d_result32, d_result8, vec, v_delta);
            checkCUDAError("after multiplying matrix");
            blob = vec;
            vec = d_result8;
            d_result8 = blob;
        }
        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);
        checkCUDAError("after loop");
        
        cudaMemcpy(h_vec, vec, sizeof(int8_t)*rows, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);

        // check if computed output is correct using hashes 
        uint32_t compute_hash = hash_int8_array(h_vec, rows);
        if (compute_hash == result_hash){
            printf("Hashes match!\n");
        }
        else{
            printf("Hashes *don't* match!\n");
            printf("[");
            for (int i=0; i<rows; i++){
                // printf("Result at index %d: %d\n", i, h_result[i]);
                printf("%d,",  h_vec[i]);
            }
            printf("]\n");
        }


        cudaEventElapsedTime(&ms1, start1, stop1);
        cudaEventElapsedTime(&ms2, start2, stop2);

        checkCUDAError("before freein the data");
        

        
        for (int k = 0; k<num_matrices; k++){
            Matrix matrix = matrices[k];
            cudaFree(matrix.d_data);
        }
        checkCUDAError("after freein the data");

        printf("Time with memcpy:    %f ms\n", ms1);
        printf("Time without memcpy: %f ms\n", ms2);
    }

    cudaFree(vec);
    delete[] h_vec;








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
