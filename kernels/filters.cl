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

// --- OPTIMIZATION: Tiled Gaussian Blur using Local Memory ---
// Uses a 16x16 workgroup size.
// Loads an 18x18 tile (16x16 + 1-pixel border) into fast local memory.
__kernel void gaussian_blur_local(__global const unsigned char* input, 
                                  __global unsigned char* output, 
                                  const int width, 
                                  const int height) 
{
    // 1. Thread IDs
    int tx = get_local_id(0); // 0-15
    int ty = get_local_id(1); // 0-15
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // 2. Define Local Memory Tile (18x18 to include halo)
    __local unsigned char tile[18][18][3]; 

    // Map local (0-15) to tile coordinates (1-16)
    int tile_x = tx + 1;
    int tile_y = ty + 1;

    // 3. Load Main Body (Center 16x16)
    if (gx < width && gy < height) {
        int idx = (gy * width + gx) * 3;
        tile[tile_y][tile_x][0] = input[idx];
        tile[tile_y][tile_x][1] = input[idx+1];
        tile[tile_y][tile_x][2] = input[idx+2];
    } else {
        tile[tile_y][tile_x][0] = 0;
        tile[tile_y][tile_x][1] = 0;
        tile[tile_y][tile_x][2] = 0;
    }

    // 4. Load Halos (Edges AND Corners)
    
    // Top Edge
    if (ty == 0) {
        int src_y = gy - 1;
        if (src_y >= 0 && gx < width) {
            int idx = (src_y * width + gx) * 3;
            tile[0][tile_x][0] = input[idx];
            tile[0][tile_x][1] = input[idx+1];
            tile[0][tile_x][2] = input[idx+2];
        }
    }
    // Bottom Edge
    if (ty == 15) {
        int src_y = gy + 1;
        if (src_y < height && gx < width) {
            int idx = (src_y * width + gx) * 3;
            tile[17][tile_x][0] = input[idx];
            tile[17][tile_x][1] = input[idx+1];
            tile[17][tile_x][2] = input[idx+2];
        }
    }
    // Left Edge
    if (tx == 0) {
        int src_x = gx - 1;
        if (src_x >= 0 && gy < height) {
            int idx = (gy * width + src_x) * 3;
            tile[tile_y][0][0] = input[idx];
            tile[tile_y][0][1] = input[idx+1];
            tile[tile_y][0][2] = input[idx+2];
        }
    }
    // Right Edge
    if (tx == 15) {
        int src_x = gx + 1;
        if (src_x < width && gy < height) {
            int idx = (gy * width + src_x) * 3;
            tile[tile_y][17][0] = input[idx];
            tile[tile_y][17][1] = input[idx+1];
            tile[tile_y][17][2] = input[idx+2];
        }
    }

    // --- CORNER LOADING (Fixed) ---
    // Top-Left Corner
    if (tx == 0 && ty == 0) {
        int src_x = gx - 1; int src_y = gy - 1;
        if (src_x >= 0 && src_y >= 0) {
            int idx = (src_y * width + src_x) * 3;
            tile[0][0][0] = input[idx]; tile[0][0][1] = input[idx+1]; tile[0][0][2] = input[idx+2];
        }
    }
    // Top-Right Corner
    if (tx == 15 && ty == 0) {
        int src_x = gx + 1; int src_y = gy - 1;
        if (src_x < width && src_y >= 0) {
            int idx = (src_y * width + src_x) * 3;
            tile[0][17][0] = input[idx]; tile[0][17][1] = input[idx+1]; tile[0][17][2] = input[idx+2];
        }
    }
    // Bottom-Left Corner
    if (tx == 0 && ty == 15) {
        int src_x = gx - 1; int src_y = gy + 1;
        if (src_x >= 0 && src_y < height) {
            int idx = (src_y * width + src_x) * 3;
            tile[17][0][0] = input[idx]; tile[17][0][1] = input[idx+1]; tile[17][0][2] = input[idx+2];
        }
    }
    // Bottom-Right Corner
    if (tx == 15 && ty == 15) {
        int src_x = gx + 1; int src_y = gy + 1;
        if (src_x < width && src_y < height) {
            int idx = (src_y * width + src_x) * 3;
            tile[17][17][0] = input[idx]; tile[17][17][1] = input[idx+1]; tile[17][17][2] = input[idx+2];
        }
    }

    // 5. Barrier Synchronization
    barrier(CLK_LOCAL_MEM_FENCE);

    // 6. Compute Convolution
    if (gx >= width || gy >= height) return;

    float sum_r = 0.0f; float sum_g = 0.0f; float sum_b = 0.0f;
    const float weights[3][3] = {
        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
        {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
        {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}
    };

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            // Read from Shared Memory (No bounds check needed now)
            unsigned char r = tile[tile_y + ky][tile_x + kx][0];
            unsigned char g = tile[tile_y + ky][tile_x + kx][1];
            unsigned char b = tile[tile_y + ky][tile_x + kx][2];

            float w = weights[ky + 1][kx + 1];
            sum_r += r * w;
            sum_g += g * w;
            sum_b += b * w;
        }
    }

    int out_idx = (gy * width + gx) * 3;
    output[out_idx]     = (unsigned char)sum_r;
    output[out_idx + 1] = (unsigned char)sum_g;
    output[out_idx + 2] = (unsigned char)sum_b;
}