

#include "h264_codec.hpp"

//    Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//    Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//    Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//    Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//    U U U U U U      V V V V V V      U V U V U V      V U V U V U
//    V V V V V V      U U U U U U      U V U V U V      V U V U V U
//    - I420 -          - YV12 -         - NV12 -         - NV21 -

#define GPU_BLOCK_THREADS  512

static dim3 gridDims(int numJobs) {
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(ceil(numJobs / (float)numBlockThreads));
}

static dim3 blockDims(int numJobs) {
    return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

    

__global__ void convert_nv12_to_bgr_float_kernel(const uint8_t* y, const uint8_t* uv, int width, int height, int linesize,  
    H264Codec::CudaNorm norm, float* dst_b, float* dst_g, float* dst_r, int edge){

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int ox = position % width;
    int oy = position / width;

    const uint8_t& yvalue = y[oy * linesize + ox];
    int offset_uv = oy / 2 * linesize + (ox / 2) * 2;
    const uint8_t& u = uv[offset_uv + 0];
    const uint8_t& v = uv[offset_uv + 1];

    float b = 1.164f * (yvalue - 16.0f) + 2.018f * (u - 128.0f);
    float g = 1.164f * (yvalue - 16.0f) - 0.813f * (v - 128.0f) - 0.391 * (u - 128.0f);
    float r = 1.164f * (yvalue - 16.0f) + 1.596f * (v - 128.0f);

    if(norm.type == H264Codec::CudaNormType::MeanSTD){
        dst_b[position] = (b * norm.scale - norm.mean[0]) / norm.std[0];
        dst_g[position] = (g * norm.scale - norm.mean[1]) / norm.std[1];
        dst_r[position] = (r * norm.scale - norm.mean[2]) / norm.std[2];
    }else if(norm.type == H264Codec::CudaNormType::ScaleAdd){
        dst_b[position] = b * norm.scale + norm.add;
        dst_g[position] = g * norm.scale + norm.add;
        dst_r[position] = r * norm.scale + norm.add;
    }else{
        dst_b[position] = b;
        dst_g[position] = g;
        dst_r[position] = r;
    }
}

namespace H264Codec{

    void convert_nv12_to_bgr_float(const uint8_t* y, const uint8_t* uv, int width, int height, int linesize, const CudaNorm& norm, float* dst, cudaStream_t stream){
        
        int total = width * height;
        auto grid = gridDims(total);
        auto block = blockDims(total);

        convert_nv12_to_bgr_float_kernel<<<grid, block, 0, stream>>>(
            y, uv, width, height, linesize, norm,
            dst + width * height * 0, 
            dst + width * height * 1, 
            dst + width * height * 2, 
            total
        );
    }
};