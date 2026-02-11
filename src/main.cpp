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
#include <cmath>

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

#define CHECK_CL(err) if (err != CL_SUCCESS) { \
    std::cerr << "OpenCL Error: " << err << " at line " << __LINE__ << std::endl; \
    exit(1); \
}

// --- Global OpenCL Handles ---
cl_context context;
cl_command_queue queue;
cl_program program;

// --- Load Kernel Source ---
std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) { 
        std::cerr << "Failed to load kernel: " << filename << std::endl; 
        exit(1); 
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// --- CPU Implementation ---
void cpu_blur(const unsigned char* input, unsigned char* output, int width, int height) {
    const float kernel[9] = { 
        1/16.0f, 2/16.0f, 1/16.0f, 
        2/16.0f, 4/16.0f, 2/16.0f, 
        1/16.0f, 2/16.0f, 1/16.0f 
    };
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int nx = std::min(std::max(x + kx, 0), width - 1);
                        int ny = std::min(std::max(y + ky, 0), height - 1);
                        sum += input[(ny * width + nx) * 3 + c] * kernel[(ky+1)*3+(kx+1)];
                    }
                }
                output[(y * width + x) * 3 + c] = (unsigned char)sum;
            }
        }
    }
}

// --- Validation Helper ---
double compareImages(const unsigned char* img1, const unsigned char* img2, size_t size) {
    double maxDiff = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = std::abs((int)img1[i] - (int)img2[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

// --- Enhanced GPU Test with Separate Timing ---
struct BenchmarkResult {
    double total_ms;      // Total time (H2D + Kernel + D2H)
    double kernel_ms;     // Pure kernel execution time
    double transfer_ms;   // Memory transfer overhead
};

BenchmarkResult runGpuTest(
    const std::string& kernelName, 
    unsigned char* host_input, 
    unsigned char* host_output,
    int w, int h, 
    bool isLocal
) {
    size_t imgSize = w * h * 3;
    cl_int err;
    BenchmarkResult result;

    Timer totalTimer;
    
    // Create buffers
    cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, imgSize, NULL, &err);
    CHECK_CL(err);
    cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, NULL, &err);
    CHECK_CL(err);

    // Transfer to device (timed separately)
    Timer transferTimer;
    err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, imgSize, host_input, 0, NULL, NULL);
    CHECK_CL(err);
    double h2d_time = transferTimer.elapsed();

    // Create kernel
    cl_kernel k = clCreateKernel(program, kernelName.c_str(), &err);
    CHECK_CL(err);

    // Set arguments
    clSetKernelArg(k, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(k, 2, sizeof(int), &w);
    clSetKernelArg(k, 3, sizeof(int), &h);

    // Configure work dimensions
    size_t localSizeArr[2] = { 16, 16 };
    size_t globalSize[2] = { 
        (size_t)((w + 15) / 16) * 16, 
        (size_t)((h + 15) / 16) * 16 
    };
    size_t* localPtr = isLocal ? localSizeArr : NULL;

    // Execute kernel with event profiling
    cl_event kernelEvent;
    err = clEnqueueNDRangeKernel(queue, k, 2, NULL, globalSize, localPtr, 0, NULL, &kernelEvent);
    CHECK_CL(err);
    clWaitForEvents(1, &kernelEvent);

    // Get kernel execution time
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    result.kernel_ms = (time_end - time_start) / 1000000.0; // Convert nanoseconds to milliseconds

    // Transfer back (timed separately)
    transferTimer.reset();
    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, imgSize, host_output, 0, NULL, NULL);
    CHECK_CL(err);
    double d2h_time = transferTimer.elapsed();

    result.total_ms = totalTimer.elapsed();
    result.transfer_ms = h2d_time + d2h_time;

    // Cleanup
    clReleaseEvent(kernelEvent);
    clReleaseKernel(k);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);

    return result;
}

int main() {
    // 1. Initial OpenCL Setup with Profiling Enabled
    cl_int err;
    cl_uint numPlatforms;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, &numPlatforms);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    // Print device info
    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    std::cout << "Using GPU: " << deviceName << std::endl;
    
    size_t localMemSize;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    std::cout << "Local Memory: " << (localMemSize / 1024) << " KB" << std::endl << std::endl;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err);
    
    // Create command queue WITH profiling enabled
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err);

    std::string sourceStr = loadKernelSource("kernels/filters.cl");
    const char* src = sourceStr.c_str();
    program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    CHECK_CL(err);
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Build Error:\n" << log.data() << std::endl;
        exit(1);
    }

    // 2. Multi-Res Configuration
    struct TestImg { std::string label; std::string path; };
    std::vector<TestImg> tests = {
        {"480p", "test_images/480p.jpg"}, 
        {"720p", "test_images/720p.jpg"},
        {"1080p", "test_images/1080p.jpg"}, 
        {"1440p", "test_images/1440p.jpg"}, 
        {"4K", "test_images/4K.jpg"}
    };

    std::ofstream csv("detailed_benchmark.csv");
    csv << "Resolution,Width,Height,CPU_ms,GPU_Std_Total_ms,GPU_Std_Kernel_ms,GPU_Std_Transfer_ms,"
        << "GPU_Local_Total_ms,GPU_Local_Kernel_ms,GPU_Local_Transfer_ms,Speedup_vs_CPU,Validation_MaxError\n";

    std::cout << "Starting Enhanced Multi-Resolution Benchmark..." << std::endl;
    std::cout << "================================================================" << std::endl;

    for (const auto& test : tests) {
        int w, h, c;
        unsigned char* data = stbi_load(test.path.c_str(), &w, &h, &c, 3);
        if (!data) { 
            std::cerr << "Skipping " << test.label << " (File not found)" << std::endl; 
            continue; 
        }

        size_t imgSize = w * h * 3;
        std::vector<unsigned char> cpu_output(imgSize);
        std::vector<unsigned char> gpu_std_output(imgSize);
        std::vector<unsigned char> gpu_local_output(imgSize);

        // --- Warm-up Pass ---
        runGpuTest("gaussian_blur", data, gpu_std_output.data(), w, h, false);
        runGpuTest("gaussian_blur_local", data, gpu_local_output.data(), w, h, true);

        const int iterations = 5;
        double cpuSum = 0;
        BenchmarkResult stdSum = {0, 0, 0};
        BenchmarkResult locSum = {0, 0, 0};

        for (int i = 0; i < iterations; i++) {
            // CPU
            Timer t_cpu;
            cpu_blur(data, cpu_output.data(), w, h);
            cpuSum += t_cpu.elapsed();

            // GPU Standard
            auto stdResult = runGpuTest("gaussian_blur", data, gpu_std_output.data(), w, h, false);
            stdSum.total_ms += stdResult.total_ms;
            stdSum.kernel_ms += stdResult.kernel_ms;
            stdSum.transfer_ms += stdResult.transfer_ms;

            // GPU Local Memory
            auto locResult = runGpuTest("gaussian_blur_local", data, gpu_local_output.data(), w, h, true);
            locSum.total_ms += locResult.total_ms;
            locSum.kernel_ms += locResult.kernel_ms;
            locSum.transfer_ms += locResult.transfer_ms;
        }

        // Average results
        double cpuAvg = cpuSum / iterations;
        BenchmarkResult stdAvg = {
            stdSum.total_ms / iterations,
            stdSum.kernel_ms / iterations,
            stdSum.transfer_ms / iterations
        };
        BenchmarkResult locAvg = {
            locSum.total_ms / iterations,
            locSum.kernel_ms / iterations,
            locSum.transfer_ms / iterations
        };

        // Validate outputs
        double stdError = compareImages(cpu_output.data(), gpu_std_output.data(), imgSize);
        double locError = compareImages(cpu_output.data(), gpu_local_output.data(), imgSize);
        double maxError = std::max(stdError, locError);

        // Calculate speedup
        double speedup = cpuAvg / locAvg.kernel_ms;

        // Print results
        std::cout << "\n" << test.label << " (" << w << "x" << h << ")" << std::endl;
        std::cout << "  CPU:              " << cpuAvg << " ms" << std::endl;
        std::cout << "  GPU Standard:" << std::endl;
        std::cout << "    Total:          " << stdAvg.total_ms << " ms" << std::endl;
        std::cout << "    Kernel Only:    " << stdAvg.kernel_ms << " ms" << std::endl;
        std::cout << "    Transfer:       " << stdAvg.transfer_ms << " ms" << std::endl;
        std::cout << "  GPU Local Memory:" << std::endl;
        std::cout << "    Total:          " << locAvg.total_ms << " ms" << std::endl;
        std::cout << "    Kernel Only:    " << locAvg.kernel_ms << " ms" << std::endl;
        std::cout << "    Transfer:       " << locAvg.transfer_ms << " ms" << std::endl;
        std::cout << "  Speedup (CPU/GPU Kernel): " << speedup << "x" << std::endl;
        std::cout << "  Validation Error: " << maxError << " (max pixel difference)" << std::endl;

        // Save to CSV
        csv << test.label << "," << w << "," << h << "," 
            << cpuAvg << "," 
            << stdAvg.total_ms << "," << stdAvg.kernel_ms << "," << stdAvg.transfer_ms << ","
            << locAvg.total_ms << "," << locAvg.kernel_ms << "," << locAvg.transfer_ms << ","
            << speedup << "," << maxError << "\n";

        // Save sample output
        if (test.label == "1080p") {
            stbi_write_jpg("output_cpu_1080p.jpg", w, h, 3, cpu_output.data(), 95);
            stbi_write_jpg("output_gpu_local_1080p.jpg", w, h, 3, gpu_local_output.data(), 95);
            std::cout << "  [Saved output images for visual inspection]" << std::endl;
        }

        stbi_image_free(data);
    }

    // 3. Cleanup
    csv.close();
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    std::cout << "\n================================================================" << std::endl;
    std::cout << "Benchmark Complete! Results saved to detailed_benchmark.csv" << std::endl;

    return 0;
}