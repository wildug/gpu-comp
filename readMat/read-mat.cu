#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#define MAX_BLOCKS 256
#define MAX_THREADS 256

void checkCUDAError(const char* msg);

__inline__ __device__ int8_t find_r(uint8_t quantile, uint8_t* cdf, int G){

    for (int8_t r=G; r>0; r--){
        if (cdf[r-1] <= quantile){
            return r-1;
        }
    }
    return -1;
};

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


class CompressedMatrix {
public:
    uint32_t rows, cols;
    float grid_spacing;
    int8_t min_value;
    uint8_t G;
    // host array pointers
    uint32_t* cursors;
    uint8_t* cdf_data;
    uint8_t* ppf_data;
    uint16_t* payload;
    uint32_t payload_size;

    // device array pointers
    uint32_t* d_cursors;
    uint8_t* d_cdf_data;
    uint8_t* d_ppf_data;
    uint16_t* d_payload;

    CompressedMatrix(uint32_t r, uint32_t c, float gs, uint32_t* cur,
                      int8_t minVal, uint8_t G, uint8_t* cdf, uint8_t* ppf, uint16_t* pay, uint32_t pay_size)
        : rows(r), cols(c), grid_spacing(gs), cursors(cur), min_value(minVal), G(G),
          cdf_data(cdf), ppf_data(ppf), payload(pay), payload_size(pay_size), d_cursors(nullptr), d_cdf_data(nullptr),
           d_ppf_data(nullptr), d_payload(nullptr){}

    __host__ static CompressedMatrix deserialize(std::ifstream& file) {
        uint32_t rows, cols;
        float grid_spacing;

        uint32_t payload_size;
        int8_t min_value;
        uint8_t G;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        file.read(reinterpret_cast<char*>(&grid_spacing), sizeof(float));
        
        // uint32_t* cursors = new uint32_t[rows];

        // non-pageable memoryP
        uint32_t* cursors;
        cudaMallocHost(&cursors, sizeof(uint32_t)*rows);

        file.read(reinterpret_cast<char*>(cursors), rows * sizeof(uint32_t));

        file.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));

        file.read(reinterpret_cast<char*>(&min_value), sizeof(int8_t));
        file.read(reinterpret_cast<char*>(&G), sizeof(uint8_t));

        uint32_t cdf_len = G + 1;
        // uint8_t* cdf_data = new uint8_t[cdf_len];
        uint8_t* cdf_data;
        cudaMallocHost(&cdf_data, sizeof(uint8_t)*cdf_len);
        file.read(reinterpret_cast<char*>(cdf_data), cdf_len);
        
        if (cdf_len % 2 == 1) {
            file.seekg(1, std::ios::cur);
        }

        // uint8_t* ppf_data = new uint8_t[256];;
        uint8_t* ppf_data;
        cudaMallocHost(&ppf_data, sizeof(uint8_t)*256);
        file.read( reinterpret_cast<char*>(ppf_data), 256);
        
        // uint16_t* payload = new uint16_t[payload_size];
        uint16_t* payload;
        cudaMallocHost(&payload, sizeof(uint16_t)*payload_size);
        file.read(reinterpret_cast<char*>(payload), payload_size * sizeof(uint16_t));
        
        if (payload_size % 2 == 1) {
            file.seekg(2, std::ios::cur);
        }

        return CompressedMatrix(rows, cols, grid_spacing, cursors, min_value, G, cdf_data, ppf_data, payload, payload_size);
    }

    float decompressAndMult(int8_t* result, int32_t* d_result32, int8_t* vector, float v_delta);
};

__global__ void decmpressAndMultiply(int32_t* dst, int8_t* vec,
     uint32_t rows, uint32_t cols, float grid_spacing,
     uint32_t* cursors, int8_t min_value, uint8_t G,
     uint8_t* cdf_data, uint8_t* ppf_data, uint16_t* payload, uint32_t payload_size
){
    unsigned int threadNo = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tId = threadIdx.x;
    // unsigned int bId = blockIdx.x;
    unsigned int blockSize = blockDim.x;
    
    
    uint32_t cursor;
    uint32_t head;
    uint8_t quantile;
    uint8_t r;
    int8_t w;
    uint8_t prob;

    __shared__ int8_t shared_vec[4096]; // TODO NOT HARDCODE THIS NUMBER
    extern __shared__ uint8_t cdf[]; // store cdf in shared memory
    __shared__ uint8_t ppf[256];


    for (int j = tId; j <cols; j+=blockSize ){
        shared_vec[j] = vec[j];
    }

    int32_t res = 0;

    // loads cdf & ppf into shared memory 
    for (int j = tId; j <G+1; j+=blockSize ){
        cdf[j] = cdf_data[j];
    }
    for (int j=tId; j< 256; j+=blockSize){
        ppf[j] = ppf_data[j];
    }

    __syncthreads();
    
    if (threadNo < rows){
        cursor = cursors[threadNo];
        head = payload[cursor] << 16 | payload[cursor+1];
        cursor +=2;
        for (int j = 0; j < cols; j++){
            quantile = head & ((1<<8)-1); // take first 8 bits of head as quantile

            r = ppf[quantile];
            // r = find_r(quantile, cdf, G);


            w = min_value + r;

            // __dp4a(srcA, srcB,c); 
            // see https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html#group__cuda__math__intrinsic__int_1ga933213059df6da2de206771f145ac2f8


            res += w * shared_vec[j]; // perform scalar addition

            prob = (cdf[r+1] - cdf[r]) & 0xFF; // modulo 2**8 to ensure it fits in a uint8
            head = (head >> 8) * prob  + (quantile -cdf[r]);
            if (head < (1<<16)){
                head = head<<16 | payload[cursor];
                cursor+=1;
            }
        }
        dst[threadNo] = res;
    }   
}
float CompressedMatrix::decompressAndMult(int8_t* d_result8, int32_t* d_result32, int8_t* vector, float v_delta){

    dim3 blockGrid(MAX_BLOCKS);
    dim3 threadBlock(MAX_THREADS);
    
    

    decmpressAndMultiply<<<blockGrid, threadBlock, (G+1)*sizeof(int8_t)>>>(d_result32, vector,
        this->rows, this->cols, this->grid_spacing,
        this-> d_cursors, this->min_value, this->G,
        this->d_cdf_data,this->d_ppf_data, this->d_payload, this->payload_size);


    float abs_max = absMaxWithThrustDevice(d_result32, this->rows);
    v_delta = abs_max / 127;
    
    int blocks = (rows+ MAX_THREADS - 1) / MAX_THREADS;
    normalizeAndRoundtoInt8<<< blocks,MAX_THREADS>>>
    (d_result32, d_result8, v_delta, this->rows);

    checkCUDAError("after kernel");
    return v_delta;
}


int main() {
    // Open the binary file
    std::string filename = "/home/wildug/RSP/myKernel/compressed_matrices.bin";
    // std::string filename = "/home/wildug/Downloads/compressed_matrices.bin";
    std::ifstream file(filename, std::ios::binary);
    
    // for timing
    float ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (!file) {
        std::cerr << "Error: Could not open file" << std::endl;
        return 1;
    }

    uint32_t num_matrices, max_word_count, len_v;
    int8_t* v0;
    int8_t* vec;
    file.read(reinterpret_cast<char*>(&num_matrices), sizeof(num_matrices));
    file.read(reinterpret_cast<char*>(&max_word_count), sizeof(max_word_count));
    file.read(reinterpret_cast<char*>(&len_v), sizeof(len_v));


    int8_t* h_v0 = new int8_t[len_v];
    file.read(reinterpret_cast<char*>(h_v0), len_v*sizeof(uint8_t));

    cudaMalloc(&v0, sizeof(int8_t)*len_v);
    cudaMemcpy(v0, h_v0,sizeof(int8_t)*len_v,  cudaMemcpyHostToDevice);

    cudaMalloc(&vec,sizeof(int8_t)*len_v);

    std::cout << "Number of matrices: " << num_matrices << std::endl;
    std::cout << "Max word-count: " << max_word_count << std::endl;
    std::cout << "len_v: " << len_v << std::endl;



    // maybe first read all files and then do the mat vec operation
    int8_t* d_result;
    int8_t* h_result;
    int rows =len_v;
    int8_t* tmp;
    std::vector<CompressedMatrix> encoded_matrices;

    for (int k = 0; k<num_matrices; k++){
        CompressedMatrix matrix = CompressedMatrix::deserialize(file);
        encoded_matrices.push_back(std::move(matrix));
    }


    file.close();

    h_result = new int8_t[rows];
    

    int max_rows;
    for (int k = 0; k<num_matrices; k++){
        if (max_rows < encoded_matrices[k].rows)              
            max_rows = encoded_matrices[k].rows;
    }


    cudaMalloc(&d_result, sizeof(int8_t)* max_rows); 

    int32_t* d_result32;
    cudaMalloc(&d_result32, sizeof(int32_t)* max_rows); 
    
    // alternatively put this inside benchmarking loop
    // MEMCPY LOOP, move cudaEventRecord above or below
    for (int k = 0; k<num_matrices; k++){
        printf("%d\n",k);
        CompressedMatrix& matrix = encoded_matrices[k];
        uint32_t* d_cursors;
        uint8_t* d_cdf_data;
        uint8_t* d_ppf_data;
        uint16_t* d_payload;

        // malloc
        checkCUDAError("before Malloc");
        cudaMalloc(&d_cursors, sizeof(uint32_t)* matrix.rows);
        cudaMalloc(&d_cdf_data, sizeof(uint8_t)*(matrix.G +1));
        cudaMalloc(&d_ppf_data, 256*sizeof(uint8_t));
        cudaMalloc(&d_payload, matrix.payload_size * sizeof(uint16_t));

        checkCUDAError("after Malloc");

        // memcpy
        cudaMemcpy(d_cursors, matrix.cursors, sizeof(uint32_t)*matrix.rows, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cdf_data, matrix.cdf_data,sizeof(uint8_t)*(matrix.G +1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ppf_data, matrix.ppf_data,sizeof(uint8_t)*256, cudaMemcpyHostToDevice);
        cudaMemcpy(d_payload, matrix.payload, matrix.payload_size * sizeof(uint16_t), cudaMemcpyHostToDevice);

        checkCUDAError("after Memcpy");


        // set the *device* pointer as object attribute
        matrix.d_cursors = d_cursors;
        matrix.d_cdf_data = d_cdf_data;
        matrix.d_ppf_data = d_ppf_data;
        matrix.d_payload =  d_payload;
    }

    for (int l=0; l< 10; l++){ // outer loop for benchmarking

        cudaMemcpy(vec, v0, sizeof(int8_t)*len_v, cudaMemcpyDeviceToDevice);

        cudaEventRecord(start);

        float v_delta = 1;

        // COMPUTE LOOP
        for (int k = 0; k<num_matrices; k++){
            CompressedMatrix& matrix = encoded_matrices[k];

            matrix.decompressAndMult(d_result, d_result32, vec, v_delta);
            checkCUDAError("after decompressing matrix");

            // to swap variables you need a third guy 'tmp'
            tmp = vec;
            vec = d_result;
            d_result = tmp;


            checkCUDAError("sizes misalign");
            rows = matrix.rows;
        }
        cudaDeviceSynchronize();



        checkCUDAError("Before Memcpy.");

        // copy 'vec' since we swapped it with d_result
        cudaMemcpy(h_result, vec, sizeof(int8_t)* rows, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("%f ms\n", ms);
        
        // alternatively put this outside of benchmarking loop
    }

    printf("freeing Memory: ");
    for (int k = 0; k<num_matrices; k++){
        CompressedMatrix matrix = encoded_matrices[k];

        cudaFreeHost(matrix.cursors);
        cudaFreeHost(matrix.cdf_data);
        cudaFreeHost(matrix.ppf_data);
        cudaFreeHost(matrix.payload);
        cudaFree(matrix.d_cursors);
        cudaFree(matrix.d_cdf_data);
        cudaFree(matrix.d_ppf_data);
        cudaFree(matrix.d_payload);
    }
    // show result
    printf("[");
    for (int i=0; i<rows; i++){
        // printf("Result at index %d: %d\n", i, h_result[i]);
        printf("%d,",  h_result[i]);
    }
    printf("]\n");
    cudaFree(d_result);
    delete[] h_v0;
    delete[] h_result;


    cudaDeviceReset();
    checkCUDAError("End of program.");
    
    return 0;
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