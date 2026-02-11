// OpenCL Kernel: Grayscale
__kernel void grayscale(__global const unsigned char* input, 
                        __global unsigned char* output, 
                        const int width, 
                        const int height) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int idx_in = (y * width + x) * 3;
    int idx_out = (y * width + x);

    unsigned char r = input[idx_in];
    unsigned char g = input[idx_in + 1];
    unsigned char b = input[idx_in + 2];

    output[idx_out] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

// OpenCL Kernel: Gaussian Blur (Standard Global Memory)
__kernel void gaussian_blur(__global const unsigned char* input, 
                            __global unsigned char* output, 
                            const int width, 
                            const int height) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    const float filter_weights[9] = {
        1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
        2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
        1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f
    };

    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;

    int channels = 3;
    int k_idx = 0;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int current_x = clamp(x + kx, 0, width - 1);
            int current_y = clamp(y + ky, 0, height - 1);

            int pixel_idx = (current_y * width + current_x) * channels;
            float weight = filter_weights[k_idx++];
            
            sum_r += input[pixel_idx]     * weight;
            sum_g += input[pixel_idx + 1] * weight;
            sum_b += input[pixel_idx + 2] * weight;
        }
    }

    int out_idx = (y * width + x) * channels;
    output[out_idx]     = (unsigned char)sum_r;
    output[out_idx + 1] = (unsigned char)sum_g;
    output[out_idx + 2] = (unsigned char)sum_b;
}

// --- CORRECTED: Tiled Gaussian Blur using Local Memory ---
// FIXES: Proper boundary handling and initialization
__kernel void gaussian_blur_local(__global const unsigned char* input, 
                                  __global unsigned char* output, 
                                  const int width, 
                                  const int height) 
{
    // Thread and group IDs
    int tx = get_local_id(0);   // 0-15
    int ty = get_local_id(1);   // 0-15
    int gx = get_global_id(0);  // Global pixel X
    int gy = get_global_id(1);  // Global pixel Y

    // Shared memory tile: Separated channels to reduce bank conflicts
    __local unsigned char tile_r[18][18];
    __local unsigned char tile_g[18][18];
    __local unsigned char tile_b[18][18];

    // Helper function for safe loading with clamping
    #define LOAD_PIXEL(tile_y_out, tile_x_out, src_y, src_x) \
    { \
        int safe_x = clamp((int)(src_x), 0, width - 1); \
        int safe_y = clamp((int)(src_y), 0, height - 1); \
        int idx = (safe_y * width + safe_x) * 3; \
        tile_r[tile_y_out][tile_x_out] = input[idx]; \
        tile_g[tile_y_out][tile_x_out] = input[idx + 1]; \
        tile_b[tile_y_out][tile_x_out] = input[idx + 2]; \
    }

    // --- Load Main Body (16x16 center) ---
    int tile_x = tx + 1;
    int tile_y = ty + 1;
    LOAD_PIXEL(tile_y, tile_x, gy, gx);

    // --- Load Halos (All threads participate) ---
    
    // Top row halo (ty == 0 loads row above)
    if (ty == 0) {
        LOAD_PIXEL(0, tile_x, gy - 1, gx);
    }
    
    // Bottom row halo (ty == 15 loads row below)
    if (ty == 15) {
        LOAD_PIXEL(17, tile_x, gy + 1, gx);
    }
    
    // Left column halo (tx == 0 loads column to left)
    if (tx == 0) {
        LOAD_PIXEL(tile_y, 0, gy, gx - 1);
    }
    
    // Right column halo (tx == 15 loads column to right)
    if (tx == 15) {
        LOAD_PIXEL(tile_y, 17, gy, gx + 1);
    }
    
    // --- Load 4 Corners (4 threads total) ---
    if (tx == 0 && ty == 0) {
        LOAD_PIXEL(0, 0, gy - 1, gx - 1);  // Top-left
    }
    if (tx == 15 && ty == 0) {
        LOAD_PIXEL(0, 17, gy - 1, gx + 1); // Top-right
    }
    if (tx == 0 && ty == 15) {
        LOAD_PIXEL(17, 0, gy + 1, gx - 1); // Bottom-left
    }
    if (tx == 15 && ty == 15) {
        LOAD_PIXEL(17, 17, gy + 1, gx + 1); // Bottom-right
    }

    // Synchronize - ensure all data is loaded before computation
    barrier(CLK_LOCAL_MEM_FENCE);

    // Boundary check for output
    if (gx >= width || gy >= height) return;

    // --- Compute 3x3 Gaussian Convolution ---
    const float w11 = 1.0f/16.0f, w12 = 2.0f/16.0f, w13 = 1.0f/16.0f;
    const float w21 = 2.0f/16.0f, w22 = 4.0f/16.0f, w23 = 2.0f/16.0f;
    const float w31 = 1.0f/16.0f, w32 = 2.0f/16.0f, w33 = 1.0f/16.0f;

    // Fully unrolled convolution
    float sum_r = 
        tile_r[tile_y-1][tile_x-1] * w11 + tile_r[tile_y-1][tile_x] * w12 + tile_r[tile_y-1][tile_x+1] * w13 +
        tile_r[tile_y  ][tile_x-1] * w21 + tile_r[tile_y  ][tile_x] * w22 + tile_r[tile_y  ][tile_x+1] * w23 +
        tile_r[tile_y+1][tile_x-1] * w31 + tile_r[tile_y+1][tile_x] * w32 + tile_r[tile_y+1][tile_x+1] * w33;

    float sum_g = 
        tile_g[tile_y-1][tile_x-1] * w11 + tile_g[tile_y-1][tile_x] * w12 + tile_g[tile_y-1][tile_x+1] * w13 +
        tile_g[tile_y  ][tile_x-1] * w21 + tile_g[tile_y  ][tile_x] * w22 + tile_g[tile_y  ][tile_x+1] * w23 +
        tile_g[tile_y+1][tile_x-1] * w31 + tile_g[tile_y+1][tile_x] * w32 + tile_g[tile_y+1][tile_x+1] * w33;

    float sum_b = 
        tile_b[tile_y-1][tile_x-1] * w11 + tile_b[tile_y-1][tile_x] * w12 + tile_b[tile_y-1][tile_x+1] * w13 +
        tile_b[tile_y  ][tile_x-1] * w21 + tile_b[tile_y  ][tile_x] * w22 + tile_b[tile_y  ][tile_x+1] * w23 +
        tile_b[tile_y+1][tile_x-1] * w31 + tile_b[tile_y+1][tile_x] * w32 + tile_b[tile_y+1][tile_x+1] * w33;

    // Write output
    int out_idx = (gy * width + gx) * 3;
    output[out_idx]     = (unsigned char)sum_r;
    output[out_idx + 1] = (unsigned char)sum_g;
    output[out_idx + 2] = (unsigned char)sum_b;

    #undef LOAD_PIXEL
}

// --- ALTERNATIVE: Simpler Local Memory Version (May be faster on some GPUs) ---
// Uses simpler loading pattern with less branching
__kernel void gaussian_blur_local_simple(__global const unsigned char* input, 
                                         __global unsigned char* output, 
                                         const int width, 
                                         const int height) 
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    __local float tile_r[18][18];
    __local float tile_g[18][18];
    __local float tile_b[18][18];

    // Each thread loads multiple pixels to fill 18x18 from 16x16 workgroup
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int load_tx = tx * 2 + dx;
            int load_ty = ty * 2 + dy;
            
            if (load_tx < 18 && load_ty < 18) {
                int load_gx = gx - tx + load_tx - 1;
                int load_gy = gy - ty + load_ty - 1;
                
                load_gx = clamp(load_gx, 0, width - 1);
                load_gy = clamp(load_gy, 0, height - 1);
                
                int idx = (load_gy * width + load_gx) * 3;
                tile_r[load_ty][load_tx] = (float)input[idx];
                tile_g[load_ty][load_tx] = (float)input[idx + 1];
                tile_b[load_ty][load_tx] = (float)input[idx + 2];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= width || gy >= height) return;

    int tile_x = tx + 1;
    int tile_y = ty + 1;

    const float w11 = 1.0f/16.0f, w12 = 2.0f/16.0f, w13 = 1.0f/16.0f;
    const float w21 = 2.0f/16.0f, w22 = 4.0f/16.0f, w23 = 2.0f/16.0f;
    const float w31 = 1.0f/16.0f, w32 = 2.0f/16.0f, w33 = 1.0f/16.0f;

    float sum_r = 
        tile_r[tile_y-1][tile_x-1] * w11 + tile_r[tile_y-1][tile_x] * w12 + tile_r[tile_y-1][tile_x+1] * w13 +
        tile_r[tile_y  ][tile_x-1] * w21 + tile_r[tile_y  ][tile_x] * w22 + tile_r[tile_y  ][tile_x+1] * w23 +
        tile_r[tile_y+1][tile_x-1] * w31 + tile_r[tile_y+1][tile_x] * w32 + tile_r[tile_y+1][tile_x+1] * w33;

    float sum_g = 
        tile_g[tile_y-1][tile_x-1] * w11 + tile_g[tile_y-1][tile_x] * w12 + tile_g[tile_y-1][tile_x+1] * w13 +
        tile_g[tile_y  ][tile_x-1] * w21 + tile_g[tile_y  ][tile_x] * w22 + tile_g[tile_y  ][tile_x+1] * w23 +
        tile_g[tile_y+1][tile_x-1] * w31 + tile_g[tile_y+1][tile_x] * w32 + tile_g[tile_y+1][tile_x+1] * w33;

    float sum_b = 
        tile_b[tile_y-1][tile_x-1] * w11 + tile_b[tile_y-1][tile_x] * w12 + tile_b[tile_y-1][tile_x+1] * w13 +
        tile_b[tile_y  ][tile_x-1] * w21 + tile_b[tile_y  ][tile_x] * w22 + tile_b[tile_y  ][tile_x+1] * w23 +
        tile_b[tile_y+1][tile_x-1] * w31 + tile_b[tile_y+1][tile_x] * w32 + tile_b[tile_y+1][tile_x+1] * w33;

    int out_idx = (gy * width + gx) * 3;
    output[out_idx]     = (unsigned char)sum_r;
    output[out_idx + 1] = (unsigned char)sum_g;
    output[out_idx + 2] = (unsigned char)sum_b;
}
