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

// --- Validation Helper ---
double compareImages(const unsigned char* img1, const unsigned char* img2, size_t size) {
    double maxDiff = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = std::abs((int)img1[i] - (int)img2[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

// --- Enhanced GPU Test with Profiling ---
struct BenchmarkResult {
    double total_ms;
    double kernel_ms;
    double transfer_ms;
};

BenchmarkResult runGpuTest(
    const std::string& kernelName, 
    unsigned char* host_input, 
    unsigned char* host_output,
    int w, int h, 
    bool isLocal,
    int* additionalArgs = nullptr,
    int numAdditionalArgs = 0
) {
    size_t imgSize = w * h * 3;
    cl_int err;
    BenchmarkResult result;

    Timer totalTimer;
    
    cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, imgSize, NULL, &err);
    CHECK_CL(err);
    cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, NULL, &err);
    CHECK_CL(err);

    Timer transferTimer;
    err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, imgSize, host_input, 0, NULL, NULL);
    CHECK_CL(err);
    double h2d_time = transferTimer.elapsed();

    cl_kernel k = clCreateKernel(program, kernelName.c_str(), &err);
    CHECK_CL(err);

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(k, 2, sizeof(int), &w);
    clSetKernelArg(k, 3, sizeof(int), &h);
    
    // Set additional arguments if provided
    for (int i = 0; i < numAdditionalArgs; i++) {
        clSetKernelArg(k, 4 + i, sizeof(int), &additionalArgs[i]);
    }

    size_t localSizeArr[2] = { 16, 16 };
    size_t globalSize[2] = { 
        (size_t)((w + 15) / 16) * 16, 
        (size_t)((h + 15) / 16) * 16 
    };
    size_t* localPtr = isLocal ? localSizeArr : NULL;

    cl_event kernelEvent;
    err = clEnqueueNDRangeKernel(queue, k, 2, NULL, globalSize, localPtr, 0, NULL, &kernelEvent);
    CHECK_CL(err);
    clWaitForEvents(1, &kernelEvent);

    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    result.kernel_ms = (time_end - time_start) / 1000000.0;

    transferTimer.reset();
    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, imgSize, host_output, 0, NULL, NULL);
    CHECK_CL(err);
    double d2h_time = transferTimer.elapsed();

    result.total_ms = totalTimer.elapsed();
    result.transfer_ms = h2d_time + d2h_time;

    clReleaseEvent(kernelEvent);
    clReleaseKernel(k);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);

    return result;
}

int main() {
    // OpenCL Setup
    cl_int err;
    cl_uint numPlatforms;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, &numPlatforms);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    std::cout << "Using GPU: " << deviceName << std::endl;
    
    size_t localMemSize;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    std::cout << "Local Memory: " << (localMemSize / 1024) << " KB" << std::endl << std::endl;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err);
    
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err);

    // Load ALL kernel files
    std::string source3x3 = loadKernelSource("kernels/filters.cl");
    std::string source7x7 = loadKernelSource("kernels/advanced_filters.cl");
    std::string combinedSource = source3x3 + "\n" + source7x7;
    
    const char* src = combinedSource.c_str();
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

    // Test Configuration
    struct TestImg { std::string label; std::string path; };
    std::vector<TestImg> tests = {
        {"720p", "test_images/720p.jpg"},
        {"1080p", "test_images/1080p.jpg"}, 
        {"4K", "test_images/4K.jpg"}
    };

    std::ofstream csv("kernel_comparison.csv");
    csv << "Resolution,Filter,Global_Kernel_ms,Local_Kernel_ms,Local_Speedup\n";

    std::cout << "GPU Image Filter Benchmark - Architecture Comparison\n";
    std::cout << "=========================================================\n\n";

    for (const auto& test : tests) {
        int w, h, c;
        unsigned char* data = stbi_load(test.path.c_str(), &w, &h, &c, 3);
        if (!data) { 
            std::cerr << "Skipping " << test.label << "\n"; 
            continue; 
        }

        size_t imgSize = w * h * 3;
        std::vector<unsigned char> gpu_std_output(imgSize);
        std::vector<unsigned char> gpu_local_output(imgSize);

        std::cout << "Testing: " << test.label << " (" << w << "x" << h << ")\n";
        std::cout << "---------------------------------------------------------\n";

        // Warm-up
        runGpuTest("gaussian_blur", data, gpu_std_output.data(), w, h, false);
        runGpuTest("gaussian_blur_local", data, gpu_local_output.data(), w, h, true);
        runGpuTest("gaussian_blur_7x7", data, gpu_std_output.data(), w, h, false);
        runGpuTest("gaussian_blur_7x7_local", data, gpu_local_output.data(), w, h, true);

        const int iterations = 5;

        // ===== 3x3 Gaussian =====
        BenchmarkResult sum_3x3_std = {0, 0, 0};
        BenchmarkResult sum_3x3_loc = {0, 0, 0};

        for (int i = 0; i < iterations; i++) {
            auto r1 = runGpuTest("gaussian_blur", data, gpu_std_output.data(), w, h, false);
            sum_3x3_std.kernel_ms += r1.kernel_ms;
            
            auto r2 = runGpuTest("gaussian_blur_local", data, gpu_local_output.data(), w, h, true);
            sum_3x3_loc.kernel_ms += r2.kernel_ms;
        }

        double avg_3x3_std = sum_3x3_std.kernel_ms / iterations;
        double avg_3x3_loc = sum_3x3_loc.kernel_ms / iterations;
        double speedup_3x3 = avg_3x3_std / avg_3x3_loc;

        std::cout << "3×3 Gaussian Blur:\n";
        std::cout << "  Global Memory:  " << avg_3x3_std << " ms\n";
        std::cout << "  Local Memory:   " << avg_3x3_loc << " ms\n";
        std::cout << "  Local Speedup:  " << speedup_3x3 << "x\n";
        if (speedup_3x3 < 1.0) {
            std::cout << "  ⚠ Local is SLOWER (expected for small kernels)\n";
        }
        std::cout << "\n";

        csv << test.label << ",3x3," << avg_3x3_std << "," << avg_3x3_loc << "," << speedup_3x3 << "\n";

        // ===== 7x7 Gaussian =====
        BenchmarkResult sum_7x7_std = {0, 0, 0};
        BenchmarkResult sum_7x7_loc = {0, 0, 0};

        for (int i = 0; i < iterations; i++) {
            auto r1 = runGpuTest("gaussian_blur_7x7", data, gpu_std_output.data(), w, h, false);
            sum_7x7_std.kernel_ms += r1.kernel_ms;
            
            auto r2 = runGpuTest("gaussian_blur_7x7_local", data, gpu_local_output.data(), w, h, true);
            sum_7x7_loc.kernel_ms += r2.kernel_ms;
        }

        double avg_7x7_std = sum_7x7_std.kernel_ms / iterations;
        double avg_7x7_loc = sum_7x7_loc.kernel_ms / iterations;
        double speedup_7x7 = avg_7x7_std / avg_7x7_loc;

        std::cout << "7×7 Gaussian Blur:\n";
        std::cout << "  Global Memory:  " << avg_7x7_std << " ms\n";
        std::cout << "  Local Memory:   " << avg_7x7_loc << " ms\n";
        std::cout << "  Local Speedup:  " << speedup_7x7 << "x\n";
        if (speedup_7x7 > 1.0) {
            std::cout << "  ✓ Local is FASTER (expected for large kernels)\n";
        }
        std::cout << "\n";

        csv << test.label << ",7x7," << avg_7x7_std << "," << avg_7x7_loc << "," << speedup_7x7 << "\n";

        std::cout << "Performance Ratio (7×7 vs 3×3):\n";
        std::cout << "  Global 7×7 is " << (avg_7x7_std / avg_3x3_std) << "× slower than 3×3\n";
        std::cout << "  Local 7×7 is " << (avg_7x7_loc / avg_3x3_loc) << "× slower than 3×3\n";
        std::cout << "\n=========================================================\n\n";

        stbi_image_free(data);
    }

    csv.close();
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    std::cout << "Benchmark Complete! Results: kernel_comparison.csv\n";
    std::cout << "\nKEY INSIGHT:\n";
    std::cout << "• 3×3 filters: Global memory wins (L2 cache is sufficient)\n";
    std::cout << "• 7×7 filters: Local memory wins (too much data for L2)\n";
    std::cout << "• This demonstrates architecture-algorithm interaction!\n";

    return 0;
}