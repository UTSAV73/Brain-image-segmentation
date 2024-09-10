import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.segmentation import chan_vese
from skimage.measure import find_contours
from skimage.filters import gaussian

#Output of your Unet/DDUnet mask goes in here
mask = cv2.imread('C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\test_Dataset\\train_masks\\10_label.tif', cv2.IMREAD_GRAYSCALE)
save_output_path='C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\test_Dataset\\train\\final.jpg'
segmentation_mask = (segmentation > 0).astype(np.uint8) * 255
cv2.imwrite(save_output_path, segmentation_mask)
print("Successfully saved output")

#Debugging
if mask is None:
    raise ValueError("Mask image not loaded correctly.")

_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

#Using cv2 to find contour
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Debugging
if len(contours) == 0:
    raise ValueError("No contours found in the mask.")

#Your brain image goes here
image = cv2.imread('C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\test_Dataset\\train\\10_input.tif')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#For debugging
if len(image_gray.shape) != 2:
    raise ValueError("Image was not converted to grayscale correctly.")
if image is None:
    raise ValueError("Main image not loaded correctly.")

# Gaussian filer to smoothen the image
sigma = 1.0  
image_smoothed = gaussian(image_gray, sigma=sigma)
image_float = img_as_float(image_smoothed)


initial_level_set = np.zeros(image_float.shape, dtype=np.float64)
initial_level_set[binary_mask > 0] = 1
initial_level_set[binary_mask == 0] = -1

#Chan-Vese segmentation, change variables like lambda 1, labda 2, max_num_iterations
'''
--> lambda1 (Inside Weight):
        This parameter controls how strongly the segmentation should favor pixels with similar intensity values inside the contour.
        A higher lambda1 means that the model will put more emphasis on keeping the object region as homogeneous as possible.
        It makes the segmentation more sensitive to intensity variations within the object.

--> lambda2 (Outside Weight):
        This parameter controls how strongly the segmentation should favor pixels with similar intensity values outside the contour. 
        A higher lambda2 makes the model more sensitive to intensity variations outside the object. 
        It helps in distinguishing the background from the object.

'''
segmentation = chan_vese(image_float, init_level_set=initial_level_set, lambda1=10, lambda2=1, tol=1e-3, max_num_iter=500)


# plt.figure(figsize=(12, 6))
# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')

# plt.subplot(1, 3, 2)
# plt.title('Initial Level Set')
# plt.imshow(initial_level_set, cmap='gray')

# plt.subplot(1, 3, 3)
# plt.title('Chan-Vese Segmentation')
# plt.imshow(segmentation, cmap='gray')

# plt.show()

#Refer to cv2 documentation for this function
contours = find_contours(segmentation, 0.5)


# Overlay the contour on the original image
image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], '-r', lw=1)

# Plot
plt.figure(figsize=(15, 7))

plt.subplot(1, 4, 1)
plt.title('Original Image (Gray)')
plt.imshow(image_gray, cmap='gray')

plt.subplot(1, 4, 2)
plt.title('Mask output from UNet')
plt.imshow(initial_level_set, cmap='gray')

plt.subplot(1, 4, 3)
plt.title('Chan-Vese Segmentation')
plt.imshow(segmentation, cmap='gray')

plt.subplot(1, 4, 4)
plt.title('Contour Overlay')
plt.imshow(image_color)
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], '-r', lw=1)
plt.axis('off')

plt.show()
