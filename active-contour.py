import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, filters
from skimage.draw import polygon_perimeter
# Implementation of Active contour, Snakes( not optimal )

# Provided functions
def create_A(a, b, N):
    row = np.r_[
        -2*a - 6*b, 
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N, N))
    for i in range(N):
        A[i] = np.roll(row, i)
    return A

def create_external_edge_force_gradients_from_img(img, sigma=30.):
    smoothed = filters.gaussian(img, sigma)
    giy, gix = np.gradient(smoothed)
    gmi = (gix**2 + giy**2)**0.5
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())
    ggmiy, ggmix = np.gradient(gmi)

    def fx(x, y):
        x[x < 0] = 0
        y[y < 0] = 0
        x[x > img.shape[1] - 1] = img.shape[1] - 1
        y[y > img.shape[0] - 1] = img.shape[0] - 1
        return ggmix[(y.round().astype(int), x.round().astype(int))]

    def fy(x, y):
        x[x < 0] = 0
        y[y < 0] = 0
        x[x > img.shape[1] - 1] = img.shape[1] - 1
        y[y > img.shape[0] - 1] = img.shape[0] - 1
        return ggmiy[(y.round().astype(int), x.round().astype(int))]

    return fx, fy

def iterate_snake(x, y, a, b, fx, fy, gamma=0.1, n_iters=10, return_all=True):
    A = create_A(a, b, x.shape[0])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)
    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x, y))
        y_ = np.dot(B, y + gamma*fy(x, y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append((x_.copy(), y_.copy()))

    if return_all:
        return snakes
    else:
        return x, y


def main(image_path, mask_path):
    image = io.imread(image_path, as_gray=True)
    mask = io.imread(mask_path, as_gray=True)
    # mask = (mask > 0).astype(np.uint8)  

    
    contours = measure.find_contours(mask, 0.5)
    largest_contour = max(contours, key=len)
    
    x_init = largest_contour[:, 1]
    y_init = largest_contour[:, 0]
    init_contour = np.array([x_init, y_init]).T

  
    fx, fy = create_external_edge_force_gradients_from_img(image)

    snakes = iterate_snake(init_contour[:, 0], init_contour[:, 1], a=0.1, b=0.005, fx=fx, fy=fy, gamma=0.05, n_iters=440)

    # Visualize results
    final_contour = snakes[-1]

    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap='gray')
    plt.plot(final_contour[0], final_contour[1], '--r', lw=1)
    plt.title('Snake (Active Contour) Result')
    plt.axis('off')
    plt.show()
    
# Path to your grayscale image and binary mask
image_path = 'C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\utils\\4036_input.jpg'
mask_path = 'C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\utils\\4036_label.jpg'

main(image_path, mask_path)

