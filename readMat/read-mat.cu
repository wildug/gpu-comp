#include <cassert>
#include <cstdio>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>

// TODO think about allocating things on gpu 

#define MAX_BLOCKS 256
#define MAX_THREADS 256

void checkCUDAError(const char* msg);

__inline__ __device__ int8_t find_r(uint8_t quantile, uint8_t* cdf, int G){
    for (int8_t r=0; r<=G; r++){
        if (cdf[r] <= quantile && quantile < cdf[r+1]){
            return r;
        }
    }
    return -1;
};



class CompressedMatrix {
public:
    uint32_t rows, cols;
    float grid_spacing;
    uint32_t* cursors;
    int8_t min_value;
    uint8_t G;
    uint8_t* cdf_data;
    uint16_t* payload;
    uint32_t payload_size;

    CompressedMatrix(uint32_t r, uint32_t c, float gs, uint32_t* cur,
                      int8_t minVal, uint8_t G, uint8_t* cdf, uint16_t* pay, uint32_t pay_size)
        : rows(r), cols(c), grid_spacing(gs), cursors(cur), min_value(minVal), G(G),
          cdf_data(cdf), payload(pay), payload_size(pay_size) {}

    ~CompressedMatrix() {
        cudaFree(cursors);
        cudaFree(cdf_data);
        cudaFree(payload);
    }

    __host__ static CompressedMatrix deserialize(std::ifstream& file) {
        uint32_t rows, cols;
        float grid_spacing;

        uint32_t payload_size;
        int8_t min_value;
        uint8_t G;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        file.read(reinterpret_cast<char*>(&grid_spacing), sizeof(float));
        
        uint32_t* cursors;
        cudaMallocManaged(&cursors, rows * sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(cursors), rows * sizeof(uint32_t));

        file.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));

        file.read(reinterpret_cast<char*>(&min_value), sizeof(int8_t));
        file.read(reinterpret_cast<char*>(&G), sizeof(uint8_t));

        uint32_t cdf_len = G + 1;
        uint8_t* cdf_data;
        cudaMallocManaged(&cdf_data, cdf_len * sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(cdf_data), cdf_len);
        
        if (cdf_len % 2 == 1) {
            file.seekg(1, std::ios::cur);
        }

        uint16_t* payload;
        cudaMallocManaged(&payload, payload_size * sizeof(uint16_t));
        file.read(reinterpret_cast<char*>(payload), payload_size * sizeof(uint16_t));
        
        if (payload_size % 2 == 1) {
            file.seekg(2, std::ios::cur);
        }

        return CompressedMatrix(rows, cols, grid_spacing, cursors, min_value, G, cdf_data, payload, payload_size);
    }

    int8_t* decompressAndMult(int8_t* result, int8_t* vector);
};

__global__ void decmpressAndMultiply(int8_t* dst, int8_t* vec,
     uint32_t rows, uint32_t cols, float grid_spacing,
     uint32_t* cursors, int8_t min_value, uint8_t G,
     uint8_t* cdf_data, uint16_t* payload, uint32_t payload_size
){
    unsigned int threadNo = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tId = threadIdx.x;
    // unsigned int bId = blockIdx.x;
    unsigned int blockSize = blockDim.x;
    
    
    uint32_t cursor;
    uint32_t head;
    uint8_t quantile;
    int8_t r;
    int8_t w;
    extern __shared__ uint8_t cdf[]; // store cdf in shared memory

    int8_t res = 0;

    // loads cdf into shared memory 
    for (int j = tId; j <=G; j+=blockSize ){
        cdf[j] = cdf_data[j];
    }

    __syncthreads();
    
    if (threadNo == 1){ // TODO CHANGED FOR DEBUGGING REMOVE
    // if (threadNo < rows){
        cursor = cursors[threadNo];
        head = payload[cursor] << 16 | payload[cursor+1];
        cursor +=2;
        for (int j = 0; j < cols; j++){
            quantile = head & ((1<<8)-1); // take first 8 bits of head as quantile
            if (threadNo == 1){
                printf("head: %u\n", head);
                printf("payload: [");
                for (uint32_t q = 0; q < 5; q++) {
                    printf("%i, ",payload[cursor+q]);
                }
                printf("]\n");
            }
            r = find_r(quantile, cdf, G);
            if (r<0){
                printf("ERRROR");
            }

            w = min_value + r;
            printf("w: %d \n",w);

            res += w * vec[j]; // perform scalar addition

            head = (head >> 8) * (cdf[r+1] - cdf[r]) + (quantile -cdf[r]);
            if (head < (2<<16)){
                printf("HIIII\n");
                head = head<<16 | payload[cursor];
                cursor+=1;
            }
            if (j==20){ //REMOVE
                printf("BREAK####\n");
                break;
            }
        }
        dst[threadNo] = res;
    }   
}
int8_t* CompressedMatrix::decompressAndMult(int8_t* result, int8_t* vector){

    dim3 blockGrid(MAX_BLOCKS);
    dim3 threadBlock(MAX_THREADS);

    decmpressAndMultiply<<<blockGrid, threadBlock, (G+1)*sizeof(int8_t)>>>(result, vector,
         this->rows, this->cols, this->grid_spacing,
          this-> cursors, this->min_value, this->G,
this->cdf_data, this->payload, this->payload_size);


    checkCUDAError("after kernel");
    return result;
}


int main() {
    // Open the binary file
    std::string filename = "/home/wildug/RSP/myKernel/compressed_matrices.bin";
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        std::cerr << "Error: Could not open file" << std::endl;
        return 1;
    }

    uint32_t num_matrices, max_word_count, len_v;
    int8_t* vec;
    file.read(reinterpret_cast<char*>(&num_matrices), sizeof(num_matrices));
    file.read(reinterpret_cast<char*>(&max_word_count), sizeof(max_word_count));
    file.read(reinterpret_cast<char*>(&len_v), sizeof(len_v));

    cudaMallocManaged(&vec, len_v * sizeof(uint8_t));

    file.read(reinterpret_cast<char*>(vec), len_v*sizeof(uint8_t));

    std::cout << "Number of matrices: " << num_matrices << std::endl;
    std::cout << "Max word-count: " << max_word_count << std::endl;
    std::cout << "len_v: " << len_v << std::endl;
    printf("v[0]: %d\n", vec[0]);
    printf("v[1]: %d\n", vec[1]);



    // maybe first read all files and then do the mat vec operation
    int8_t* d_result;
    int rows;
    for (int k = 0; k<num_matrices; k++){

        CompressedMatrix matrix = CompressedMatrix::deserialize(file);
        cudaMalloc(&d_result, sizeof(uint8_t)* matrix.rows);

        matrix.decompressAndMult(d_result, vec);
        checkCUDAError("after decompress");
        vec = d_result;

        // std::cout << "Rows: " << matrix.rows << std::endl;
        // std::cout << "Columns: " << matrix.cols << std::endl;
        // printf("Min value %i\n", matrix.min_value);
        assert(matrix.rows == 1024);
        assert(matrix.cols == 1024);
        rows = matrix.rows;

        break; //REMOVE
    }

    // Close the file
    file.close();

    int8_t* h_result = new int8_t[rows];

    cudaMemcpy(h_result, d_result, sizeof(int8_t)* rows, cudaMemcpyDeviceToHost);

    // show result
    // for (int i=0; i<10; i++){
    //     printf("Result at index %d: %d\n", i, h_result[i]);
    // }
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