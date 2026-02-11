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

// OpenCL Kernel: Gaussian Blur (3x3)
__kernel void gaussian_blur(__global const unsigned char* input, 
                            __global unsigned char* output, 
                            const int width, 
                            const int height) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // RENAME 'kernel' -> 'filter_weights'
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

            // RENAME HERE TOO
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