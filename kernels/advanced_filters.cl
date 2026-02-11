// 7×7 Gaussian Blur Kernel (Global Memory)
// This will be SLOW - demonstrates when global memory struggles
__kernel void gaussian_blur_7x7(__global const unsigned char* input, 
                                __global unsigned char* output, 
                                const int width, 
                                const int height) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // 7×7 Gaussian weights (normalized)
    const float weights[49] = {
        0.000036f, 0.000363f, 0.001446f, 0.002291f, 0.001446f, 0.000363f, 0.000036f,
        0.000363f, 0.003676f, 0.014662f, 0.023226f, 0.014662f, 0.003676f, 0.000363f,
        0.001446f, 0.014662f, 0.058488f, 0.092651f, 0.058488f, 0.014662f, 0.001446f,
        0.002291f, 0.023226f, 0.092651f, 0.146768f, 0.092651f, 0.023226f, 0.002291f,
        0.001446f, 0.014662f, 0.058488f, 0.092651f, 0.058488f, 0.014662f, 0.001446f,
        0.000363f, 0.003676f, 0.014662f, 0.023226f, 0.014662f, 0.003676f, 0.000363f,
        0.000036f, 0.000363f, 0.001446f, 0.002291f, 0.001446f, 0.000363f, 0.000036f
    };

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    int k_idx = 0;

    // 49 iterations - each accessing global memory
    for (int ky = -3; ky <= 3; ++ky) {
        for (int kx = -3; kx <= 3; ++kx) {
            int nx = clamp(x + kx, 0, width - 1);
            int ny = clamp(y + ky, 0, height - 1);
            
            int idx = (ny * width + nx) * 3;
            float w = weights[k_idx++];
            
            sum_r += input[idx]     * w;
            sum_g += input[idx + 1] * w;
            sum_b += input[idx + 2] * w;
        }
    }

    int out_idx = (y * width + x) * 3;
    output[out_idx]     = (unsigned char)sum_r;
    output[out_idx + 1] = (unsigned char)sum_g;
    output[out_idx + 2] = (unsigned char)sum_b;
}

// 7×7 Gaussian Blur with Local Memory Optimization
// This should be 2-3× faster than global memory version!
__kernel void gaussian_blur_7x7_local(__global const unsigned char* input, 
                                      __global unsigned char* output, 
                                      const int width, 
                                      const int height) 
{
    int tx = get_local_id(0);   // 0-15
    int ty = get_local_id(1);   // 0-15
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // 22×22 tile (16 + 2×3 halo on each side)
    __local float tile_r[22][22];
    __local float tile_g[22][22];
    __local float tile_b[22][22];

    // Cooperative loading - each thread loads multiple pixels
    // Strategy: each thread loads a 2×2 block (covers 32×32 / 256 threads = 4 pixels/thread)
    // We need 22×22 = 484 pixels total
    
    for (int load_idx = tx * 16 + ty; load_idx < 484; load_idx += 256) {
        int load_tx = load_idx % 22;
        int load_ty = load_idx / 22;
        
        int load_gx = gx - tx + load_tx - 3;
        int load_gy = gy - ty + load_ty - 3;
        
        load_gx = clamp(load_gx, 0, width - 1);
        load_gy = clamp(load_gy, 0, height - 1);
        
        int idx = (load_gy * width + load_gx) * 3;
        tile_r[load_ty][load_tx] = (float)input[idx];
        tile_g[load_ty][load_tx] = (float)input[idx + 1];
        tile_b[load_ty][load_tx] = (float)input[idx + 2];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= width || gy >= height) return;

    int tile_x = tx + 3;
    int tile_y = ty + 3;

    // 7×7 Gaussian weights
    const float weights[49] = {
        0.000036f, 0.000363f, 0.001446f, 0.002291f, 0.001446f, 0.000363f, 0.000036f,
        0.000363f, 0.003676f, 0.014662f, 0.023226f, 0.014662f, 0.003676f, 0.000363f,
        0.001446f, 0.014662f, 0.058488f, 0.092651f, 0.058488f, 0.014662f, 0.001446f,
        0.002291f, 0.023226f, 0.092651f, 0.146768f, 0.092651f, 0.023226f, 0.002291f,
        0.001446f, 0.014662f, 0.058488f, 0.092651f, 0.058488f, 0.014662f, 0.001446f,
        0.000363f, 0.003676f, 0.014662f, 0.023226f, 0.014662f, 0.003676f, 0.000363f,
        0.000036f, 0.000363f, 0.001446f, 0.002291f, 0.001446f, 0.000363f, 0.000036f
    };

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    int k_idx = 0;

    // Now all reads are from fast local memory!
    for (int ky = -3; ky <= 3; ++ky) {
        for (int kx = -3; kx <= 3; ++kx) {
            float w = weights[k_idx++];
            sum_r += tile_r[tile_y + ky][tile_x + kx] * w;
            sum_g += tile_g[tile_y + ky][tile_x + kx] * w;
            sum_b += tile_b[tile_y + ky][tile_x + kx] * w;
        }
    }

    int out_idx = (gy * width + gx) * 3;
    output[out_idx]     = (unsigned char)sum_r;
    output[out_idx + 1] = (unsigned char)sum_g;
    output[out_idx + 2] = (unsigned char)sum_b;
}

// Box Blur - Separable Filter Demonstration
// This is perfect for local memory because of the two-pass approach
__kernel void box_blur_horizontal(__global const unsigned char* input, 
                                  __global unsigned char* output, 
                                  const int width, 
                                  const int height,
                                  const int radius)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    int count = 0;

    for (int kx = -radius; kx <= radius; ++kx) {
        int nx = clamp(x + kx, 0, width - 1);
        int idx = (y * width + nx) * 3;
        sum_r += input[idx];
        sum_g += input[idx + 1];
        sum_b += input[idx + 2];
        count++;
    }

    int out_idx = (y * width + x) * 3;
    output[out_idx]     = (unsigned char)(sum_r / count);
    output[out_idx + 1] = (unsigned char)(sum_g / count);
    output[out_idx + 2] = (unsigned char)(sum_b / count);
}

__kernel void box_blur_vertical(__global const unsigned char* input, 
                                __global unsigned char* output, 
                                const int width, 
                                const int height,
                                const int radius)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    int count = 0;

    for (int ky = -radius; ky <= radius; ++ky) {
        int ny = clamp(y + ky, 0, height - 1);
        int idx = (ny * width + x) * 3;
        sum_r += input[idx];
        sum_g += input[idx + 1];
        sum_b += input[idx + 2];
        count++;
    }

    int out_idx = (y * width + x) * 3;
    output[out_idx]     = (unsigned char)(sum_r / count);
    output[out_idx + 1] = (unsigned char)(sum_g / count);
    output[out_idx + 2] = (unsigned char)(sum_b / count);
}
