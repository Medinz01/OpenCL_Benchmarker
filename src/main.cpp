#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <cstring> // for memcpy

// --- Helper: Timer ---
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// --- Helper: OpenCL Error Checker ---
#define CHECK_CL(err) if (err != CL_SUCCESS) { \
    std::cerr << "OpenCL Error: " << err << " at line " << __LINE__ << std::endl; \
    exit(1); \
}

std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) { std::cerr << "Failed to load kernel: " << filename << std::endl; exit(1); }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// --- CPU ALGORITHMS ---
void cpu_grayscale(const unsigned char* input, unsigned char* output, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        int idx = i * 3;
        output[i] = (unsigned char)(0.299f * input[idx] + 0.587f * input[idx+1] + 0.114f * input[idx+2]);
    }
}

void cpu_blur(const unsigned char* input, unsigned char* output, int width, int height) {
    const float kernel[9] = {
        1/16.0f, 2/16.0f, 1/16.0f,
        2/16.0f, 4/16.0f, 2/16.0f,
        1/16.0f, 2/16.0f, 1/16.0f
    };
    int channels = 3;
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                int k_idx = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int p_idx = ((y + ky) * width + (x + kx)) * channels + c;
                        sum += input[p_idx] * kernel[k_idx++];
                    }
                }
                output[(y*width+x)*channels + c] = (unsigned char)sum;
            }
        }
    }
}

int main() {
    // 1. SETUP
    std::string inputPath = "input.jpg";
    int width, height, channels;
    unsigned char* host_input = stbi_load(inputPath.c_str(), &width, &height, &channels, 0);
    if (!host_input) { std::cerr << "Load failed. Check input.jpg" << std::endl; return 1; }
    
    std::cout << "Benchmarking Image: " << width << "x" << height << std::endl;
    std::ofstream csv("benchmark_results.csv");
    csv << "Algorithm,Device,Time_ms\n";

    // Allocations
    std::vector<unsigned char> cpu_gray_out(width * height);
    std::vector<unsigned char> cpu_blur_out(width * height * 3);
    std::vector<unsigned char> gpu_out(width * height * 3);
    std::vector<unsigned char> gpu_gray_out(width * height);

    // 2. CPU BENCHMARKS
    Timer t;
    std::cout << "Running CPU Grayscale..." << std::endl;
    cpu_grayscale(host_input, cpu_gray_out.data(), width, height);
    double cpuGrayTime = t.elapsed();
    csv << "Grayscale,CPU," << cpuGrayTime << "\n";

    t.reset();
    std::cout << "Running CPU Blur..." << std::endl;
    cpu_blur(host_input, cpu_blur_out.data(), width, height);
    double cpuBlurTime = t.elapsed();
    csv << "Gaussian Blur,CPU," << cpuBlurTime << "\n";

    // 3. GPU SETUP
    cl_int err;
    cl_uint numPlatforms;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, &numPlatforms);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    
    std::string sourceStr = loadKernelSource("kernels/filters.cl");
    const char* src = sourceStr.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    // Check for build errors
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        std::cerr << "Build Error Log:\n" << log << std::endl;
        exit(1);
    }

    cl_kernel k_gray = clCreateKernel(program, "grayscale", &err);
    CHECK_CL(err);
    cl_kernel k_blur = clCreateKernel(program, "gaussian_blur", &err);
    CHECK_CL(err);

    size_t imgSize = width * height * channels;
    size_t graySize = width * height;

    // --- STANDARD GPU BENCHMARK (Baseline) ---
    t.reset();
    cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgSize, host_input, &err);
    cl_mem d_out_gray = clCreateBuffer(context, CL_MEM_WRITE_ONLY, graySize, NULL, &err);
    
    clSetKernelArg(k_gray, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(k_gray, 1, sizeof(cl_mem), &d_out_gray);
    clSetKernelArg(k_gray, 2, sizeof(int), &width);
    clSetKernelArg(k_gray, 3, sizeof(int), &height);
    
    size_t globalSize[2] = { (size_t)width, (size_t)height };
    clEnqueueNDRangeKernel(queue, k_gray, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_gray, CL_TRUE, 0, graySize, gpu_gray_out.data(), 0, NULL, NULL);
    double gpuGrayTime = t.elapsed();
    csv << "Grayscale,GPU (Standard)," << gpuGrayTime << "\n";
    std::cout << "GPU Grayscale (Standard) done: " << gpuGrayTime << " ms" << std::endl;

    // GPU Blur
    t.reset();
    cl_mem d_out_blur = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, NULL, &err);
    
    clSetKernelArg(k_blur, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(k_blur, 1, sizeof(cl_mem), &d_out_blur);
    clSetKernelArg(k_blur, 2, sizeof(int), &width);
    clSetKernelArg(k_blur, 3, sizeof(int), &height);
    
    clEnqueueNDRangeKernel(queue, k_blur, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_blur, CL_TRUE, 0, imgSize, gpu_out.data(), 0, NULL, NULL);
    double gpuBlurTime = t.elapsed();
    csv << "Gaussian Blur,GPU (Standard)," << gpuBlurTime << "\n";
    std::cout << "GPU Blur (Standard) done: " << gpuBlurTime << " ms" << std::endl;


    // --- OPTIMIZATION 1: PINNED MEMORY (Zero-Copy) ---
    t.reset();
    cl_mem pinned_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, imgSize, NULL, &err);
    CHECK_CL(err);
    unsigned char* pinned_ptr_in = (unsigned char*)clEnqueueMapBuffer(queue, pinned_in, CL_TRUE, CL_MAP_WRITE, 0, imgSize, 0, NULL, NULL, &err);
    CHECK_CL(err);
    std::memcpy(pinned_ptr_in, host_input, imgSize);
    clEnqueueUnmapMemObject(queue, pinned_in, pinned_ptr_in, 0, NULL, NULL);

    cl_mem pinned_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, graySize, NULL, &err);
    CHECK_CL(err);

    clSetKernelArg(k_gray, 0, sizeof(cl_mem), &pinned_in);
    clSetKernelArg(k_gray, 1, sizeof(cl_mem), &pinned_out);
    clEnqueueNDRangeKernel(queue, k_gray, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    unsigned char* pinned_ptr_out = (unsigned char*)clEnqueueMapBuffer(queue, pinned_out, CL_TRUE, CL_MAP_READ, 0, graySize, 0, NULL, NULL, &err);
    clEnqueueUnmapMemObject(queue, pinned_out, pinned_ptr_out, 0, NULL, NULL);
    
    double pinnedTime = t.elapsed();
    csv << "Grayscale,GPU (Pinned)," << pinnedTime << "\n";
    std::cout << "GPU Grayscale (Pinned) done: " << pinnedTime << " ms" << std::endl;

    // --- OPTIMIZATION: LOCAL MEMORY TILING (FAIR SYSTEM TIME) ---
    // We must measure: Upload -> Kernel -> Download
    t.reset();

    // 1. Create FRESH input buffer (This triggers the Host->Device Upload)
    // Using COPY_HOST_PTR forces the copy to happen now.
    cl_mem d_in_local = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgSize, host_input, &err);
    CHECK_CL(err);

    // 2. Create FRESH output buffer
    cl_mem d_out_local = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, NULL, &err);
    CHECK_CL(err);

    // 3. Setup Kernel
    cl_kernel k_blur_local = clCreateKernel(program, "gaussian_blur_local", &err);
    CHECK_CL(err);

    clSetKernelArg(k_blur_local, 0, sizeof(cl_mem), &d_in_local);
    clSetKernelArg(k_blur_local, 1, sizeof(cl_mem), &d_out_local);
    clSetKernelArg(k_blur_local, 2, sizeof(int), &width);
    clSetKernelArg(k_blur_local, 3, sizeof(int), &height);

    // 4. Work Group Sizing
    size_t localSize[2] = {16, 16};
    size_t globalSizeRounded[2] = {
        (size_t)((width + 15) / 16) * 16, 
        (size_t)((height + 15) / 16) * 16
    };

    // 5. Execute
    CHECK_CL(clEnqueueNDRangeKernel(queue, k_blur_local, 2, NULL, globalSizeRounded, localSize, 0, NULL, NULL));
    
    // 6. Read Back (Device -> Host Download)
    // The timer stops only after we have the data back in RAM.
    CHECK_CL(clEnqueueReadBuffer(queue, d_out_local, CL_TRUE, 0, imgSize, gpu_out.data(), 0, NULL, NULL));
    
    double localTime = t.elapsed();
    
    csv << "Gaussian Blur,GPU (Local Mem)," << localTime << "\n";
    std::cout << "GPU Blur (Local Mem) done: " << localTime << " ms" << std::endl;

    // Cleanup local resources
    clReleaseMemObject(d_in_local);
    clReleaseMemObject(d_out_local);
    clReleaseKernel(k_blur_local);

    // --- CLEANUP ---
    csv.close();
    std::cout << "Benchmark Complete! Results saved." << std::endl;

    clReleaseMemObject(d_in); clReleaseMemObject(d_out_gray); clReleaseMemObject(d_out_blur);
    clReleaseMemObject(pinned_in); clReleaseMemObject(pinned_out);
    clReleaseKernel(k_gray); clReleaseKernel(k_blur); clReleaseKernel(k_blur_local);
    clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
    stbi_image_free(host_input);
    
    return 0;
}