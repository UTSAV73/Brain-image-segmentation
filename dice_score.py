from PIL import Image
import numpy as np

 #convert image to ndarray
def load_image_as_binary(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('L')  
    img_array = np.array(img)
    binary_array = (img_array > 0).astype(int) 
    return binary_array

def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the Dice coefficient between two binary masks.
    
    Args:
    - mask1 (np.ndarray): Binary mask 1 (numpy array with 0s and 1s)
    - mask2 (np.ndarray): Binary mask 2 (numpy array with 0s and 1s)
    
    Returns:
    - float: Dice coefficient
    """  
    intersection = np.sum(mask1 * mask2)
    sum_of_sizes = np.sum(mask1) + np.sum(mask2)
    dice = 2. * intersection / sum_of_sizes if sum_of_sizes != 0 else 1.0
    return dice

 # Mask1 could be your ground truth and mask2 could be your output
 
mask1_path ='C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\3022_label.jpg'
mask2_path = 'C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\pred_mask.jpg'

mask1 = load_image_as_binary(mask1_path)
mask2 = load_image_as_binary(mask2_path)

dice_score = dice_coefficient(mask1, mask2)
print(f"Dice coefficient: {dice_score:.4f}")

