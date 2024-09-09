import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
from data_maker import datasetBuilder
from unet import UNet
from DDUNet import DDUNet
import numpy as np
###################################


                 # Uncomment and comment the functions and lines accordingly to test on a dataset and on a single image




def post_process_output(output):
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()  
    
    # Example: Assume output is in range [0, 1] and needs to be scaled to [0, 255]
    output = (output * 255).astype(np.uint8)
    
    # If the output is a grayscale image (single channel), ensure it has 2D shape
    if len(output.shape) == 3 and output.shape[2] == 1:
        output = output.squeeze(axis=-1)
    
    return output

def save_image(output, path):
    """
    Save an image to a file.
    
    Parameters:
        output (numpy.ndarray): The processed image.
        path (str): The path to save the image.
    """
    # Save the image using OpenCV
    success = cv2.imwrite(path, output)
    # Debugging
    if success:
        print(f"Image successfully saved to {path}")
    else:
        print(f"Failed to save the image to {path}")
#########################

                                      # Use below function to test on a dataset
'''
def pred_show_image_grid(data_path, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = CarvanaDataset(data_path, test=True)
    images = []
    orig_masks = []
    pred_masks = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0]=0
        pred_mask[pred_mask > 0]=1

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

    images.extend(orig_masks)
    images.extend(pred_masks)
    fig = plt.figure()
    for i in range(1, 3*len(image_dataset)+1):
       fig.add_subplot(3, len(image_dataset), i)
       plt.imshow(images[i-1], cmap="gray")
    plt.show()
'''

# This is for masking a single image 

def single_image_inference(image_pth, model_pth, device):
    model = DDUNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)
   
    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0]=0
    pred_mask[pred_mask > 0]=1
    output=post_process_output(pred_mask)
    # Add Path to save image
    save_image(output,"C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\utils\\pred_mask.jpg")
    fig = plt.figure()
    for i in range(1, 3): 
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")
    plt.show()


if __name__ == "__main__":
    SINGLE_IMG_PATH = "C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\utils\\4036_input.jpg"
    ### DATA_PATH = "./data"
    MODEL_PATH = "C:\\Users\\Utsav\OneDrive\\Desktop\\research\\diy\\unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ### pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)
    