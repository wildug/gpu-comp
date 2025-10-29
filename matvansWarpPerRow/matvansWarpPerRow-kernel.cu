#define WARP_SIZE 32

template<int Capacity>
struct ThreadQueue {
    uint16_t data[Capacity];
    int head;  // index for dequeue
    int tail;  // index for enqueue
    int size;
    int4* data4;

    __device__ void init() {
        head = 0;
        tail = 0;
        size = 0;
    }

    __device__ bool isEmpty() const {
        return size == 0;
    }

    __device__ bool isFull() const {
        return size > (Capacity-8);
    }

    // enqueue of int4
    __device__ bool enqueue(int4 val) {
        if (isFull()) return false;

        data4 = reinterpret_cast<int4*>(&data[tail]);
        data4[0] = val;
        tail = (tail + 8) % Capacity;
        size+=8;
        return true;
    }

    // dequeue of uint16_t: direct access
    __device__ bool dequeue(uint16_t &out) {
        if (isEmpty()) return false;
        out = data[head];
        head = (head + 1) % Capacity;
        size--;
        return true;
    }
};

__global__ void decmpressAndMultiply(int32_t* dst, int8_t* vec,
     uint32_t rows, uint32_t cols, float grid_spacing,
     const uint32_t* cursors, int8_t min_value, uint8_t G,
     const uint8_t* cdf_data, const uint8_t* ppf_data, uint16_t* payload, uint32_t payload_size
){
    unsigned int threadNo = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tId = threadIdx.x;
    unsigned int lane = tId % 32;
    // one warpPerRow means warpId == row
    unsigned int warpId = threadNo / 32;
    // unsigned int bId = blockIdx.x;
    unsigned int blockSize = blockDim.x;
    
    
    uint32_t cursor;
    uint32_t head;
    uint8_t quantile;
    uint8_t r;
    int32_t w;
    uint8_t prob;
    uint16_t word;
    int8_t* bytes;

    extern __shared__ int8_t shared_vec[]; 
    __shared__ uint8_t cdf[257]; // store cdf in shared memory
    __shared__ uint8_t ppf[256];





    int4* shared_vec4 = reinterpret_cast<int4*>(shared_vec);
    int4* vec4 = reinterpret_cast<int4*>(vec);
    for (int j = tId; j <(cols/16); j+=blockSize ){
        shared_vec4[j] =  vec4[j];
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
    
    ThreadQueue<16> q;
    q.init();
    cursor = cursors[warpId];
    head = payload[cursor+lane*8] << 16 | payload[cursor+lane*8+1];
    int4* payload4 = reinterpret_cast<int4*>(&payload[cursor]);
    int cursor4 = lane;

    int shared_vec_cursor = lane;
    int4 v4;
    bool remmi = q.enqueue(payload4[cursor4]);
    // dequeue 2 words since first two words are used for the coder head
    q.dequeue(word);
    q.dequeue(word);
    cursor4+=32;

    int8_t w0, w1, w2, w3;
    for (int j = 0; j < cols/WARP_SIZE; j++){
            // does not diverege across warp
            if (j%16==0){
                v4 = shared_vec4[shared_vec_cursor];
                shared_vec_cursor+=WARP_SIZE;
                bytes = reinterpret_cast<int8_t*>(&v4);
            }


            quantile = head & ((1<<8)-1); // take first 8 bits of head as quantile

            r = ppf[quantile];
            // r = find_r(quantile, cdf, G);


            w0 = min_value + r;

            // __dp4a(srcA, srcB,c); 
            // see https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html#group__cuda__math__intrinsic__int_1ga933213059df6da2de206771f145ac2f8



            res += w0 * bytes[j%16]; // perform scalar addition

            prob = (cdf[r+1] - cdf[r]) & 0xFF; // modulo 2**8 to ensure it fits in a uint8
            head = (head >> 8) * prob  + (quantile -cdf[r]);
            // if (warpId == 136 && lane==16){
            //     // printf("decoded %d,   col: %d, lane: %d \n", w0, j, lane);
            //     printf("%d, \t %d \n ",  bytes[j%16],w0);
            // }

            bool this_lane_needs_refill = q.isEmpty();
            unsigned int any_result = __any_sync(0xFFFFFFFF, this_lane_needs_refill);
            // does not diverege across warp
            if (any_result){
                bool s = q.enqueue(payload4[cursor4]);
                if (s){
                    cursor4+=32;
                }
                // XOR
                // if (!q.enqueue(payload4[cursor4])){
                //     printf("Wir haben kein remmydemmi!!\n");
                // };
            }
            
            // diverege across warp!
            if (head < (1<<16)){
                q.dequeue(word);
                // XOR
                // if(!q.dequeue(word)){
                //     printf("Wir haben kein remmydemmi!!\n");
                // }
                head = head<<16 | word;
            }
        
        }   
        __syncwarp();


        // Warp reduction (sum partials across lanes)
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            res += __shfl_down_sync(0xffffffff, res, offset);
        }
        __syncwarp();


        // Lane 0 writes the row result
        if (lane == 0){
            // printf("warpId: %d, res: %d \n", warpId, res);
            dst[warpId] = res;
        }

}