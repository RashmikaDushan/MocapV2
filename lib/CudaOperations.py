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

@cuda.jit
def demosaic_kernel(bayer_img, rgb_img):
    y, x = cuda.grid(2)
    height, width = bayer_img.shape

    def get_pixel(xx, yy):
        if 0 <= xx < width and 0 <= yy < height:
            return bayer_img[yy, xx]
        return 0

    if y >= height or x >= width:
        return

    r = g = b = 0

    if (y % 2 == 0) and (x % 2 == 0):
        # Blue pixel
        b = get_pixel(x, y)
        g = (get_pixel(x-1, y) + get_pixel(x+1, y) + get_pixel(x, y-1) + get_pixel(x, y+1)) // 4
        r = (get_pixel(x-1, y-1) + get_pixel(x+1, y-1) + get_pixel(x-1, y+1) + get_pixel(x+1, y+1)) // 4
    elif (y % 2 == 0) and (x % 2 == 1):
        # Green pixel on blue row
        g = get_pixel(x, y)
        b = (get_pixel(x-1, y) + get_pixel(x+1, y)) // 2
        r = (get_pixel(x, y-1) + get_pixel(x, y+1)) // 2
    elif (y % 2 == 1) and (x % 2 == 0):
        # Green pixel on red row
        g = get_pixel(x, y)
        r = (get_pixel(x-1, y) + get_pixel(x+1, y)) // 2
        b = (get_pixel(x, y-1) + get_pixel(x, y+1)) // 2
    else:
        # Red pixel
        r = get_pixel(x, y)
        g = (get_pixel(x-1, y) + get_pixel(x+1, y) + get_pixel(x, y-1) + get_pixel(x, y+1)) // 4
        b = (get_pixel(x-1, y-1) + get_pixel(x+1, y-1) + get_pixel(x-1, y+1) + get_pixel(x+1, y+1)) // 4

    rgb_img[y, x, 0] = b
    rgb_img[y, x, 1] = g
    rgb_img[y, x, 2] = r


def fast_cuda_demosaic(bayer_img: np.ndarray) -> np.ndarray:
    assert len(bayer_img.shape) == 2, "Expected a single-channel Bayer image (grayscale)"

    height, width = bayer_img.shape
    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)

    d_bayer = cuda.to_device(bayer_img)
    d_rgb = cuda.to_device(rgb_img)

    threadsperblock = (16, 16)
    blockspergrid_x = (width + 15) // 16
    blockspergrid_y = (height + 15) // 16
    blockspergrid = (blockspergrid_y, blockspergrid_x)

    demosaic_kernel[blockspergrid, threadsperblock](d_bayer, d_rgb)

    return d_rgb.copy_to_host()
