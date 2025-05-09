import numpy as np
import cv2
from numba import cuda, float32

@cuda.jit
def blur_kernel(input_img, output_img, kernel_size):
    i, j = cuda.grid(2)
    height, width = input_img.shape

    k = kernel_size // 2
    sum = 0.0
    count = 0

    if i < height and j < width:
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                ni = i + di
                nj = j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    sum += input_img[ni, nj]
                    count += 1
        output_img[i, j] = sum / count

def fast_cuda_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    assert len(image.shape) == 2, "Only grayscale images supported"
    height, width = image.shape

    input_img = image.astype(np.float32)
    output_img = np.zeros_like(input_img)

    d_input = cuda.to_device(input_img)
    d_output = cuda.to_device(output_img)

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_y = (height + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid = (blockspergrid_y, blockspergrid_x)

    blur_kernel[blockspergrid, threadsperblock](d_input, d_output, kernel_size)

    return d_output.copy_to_host().astype(np.uint8)
