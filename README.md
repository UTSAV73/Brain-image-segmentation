
# Early detection of Parksinson's disease using MRI and SPECT scans with scarce dataset.

The follow model uses a UNet/DDUNet architecture to segement [Subtantia Nigra](https://en.wikipedia.org/wiki/Substantia_nigra) from the brain scans to produce binary masks which is then classified as Parkinson and Non-Parksinson by a RESNet classifier.



## Installation

Clone this project

```bash
  git clone https://github.com/UTSAV73/Brain-image-segmentation.git
```
Make a new virtual env accordingly (use conda) and install requirements from requirements.txt

```bash
  pip install requirements.txt
```
Make a new directory for dataset and inside, make 2 other directories for training images and another for their corresponding masks. Same should be done for testing. It should look like:
```bash
Brain-image-segmentation/
│
├── dataset/
│   ├── train/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── train_masks/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── test/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── test_masks/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
|
```

## Note
 - # Since the code uses absolute paths, follow the comments and change the paths accordingly.
  

## Usage

- Run the train.py to train and adjust EPOCHS, batch-size etc in the code. You can choose the UNET or DDUNET model accodingly.
- Then use test.py which has testing with dataset and single inference function out of which either can be commented out to use the other accordingly.
- Run the classifiertrain.py to train the RESNet with your binary masks. Classifier diectory should have 2 sub directories:
```bash
classifier/
│
├── controlSEWEDD/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── PD/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── classifiertrain.py
└── classifier.py
```

- The output of the test.py inference is saved which can then be fed into the classifier.py to classify it as PD and NON-PD (can be SWEDD or CONTROL).


    
## Inference
Segementation of brain cancer:
<img width="474" alt="image" src="https://github.com/user-attachments/assets/268f71d4-f089-4b88-8b65-8362768b5343">

Segemention of Subtantia Nigra:
![image](https://github.com/user-attachments/assets/6bd40394-f576-4275-9db7-9c2bfb960e48)



## Features and Optimizations

- Using Image Segmentation models like Dense-Diluted UNet which can work on noisy MRI images with low resolutions.
- Use of Chan-Vese active contour segmentation model to refine the segmentrd output and increase the accuracy of classifier.
-  Use of RESNet trained on custom input instead of transfer learning.
- Low GPU memory usage and fast training time.
- Easily trainable on local GPU/TPU/CPU.
- Illustrative and detailed output for inference.


## Future Work

- Trainable parameters for Chan-Vese (Lambda1, Lambda2 and PHI)
    - Lambda1: Weight for the average intensity inside the contour.
    - Lambda2: Weight for the average intensity outside the contour.
    - PHI: The initial level set function, often initialized to a signed distance function from the contour.
- Build and deploy as an APP
- Use Attention-based Vision Transformer to improve accuracy
- Compile a local dataset


## Research and Exploration

https://docs.google.com/document/d/1H5xZ2pnQuOh_wnTkkgqBvmwbd8KZk3fHt4BbXgSZ0sw/edit?usp=sharing
