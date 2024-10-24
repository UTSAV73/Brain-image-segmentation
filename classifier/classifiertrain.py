import os
import numpy as np
import cv2
import glob
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (512, 512)  
BATCH_SIZE = 2
EPOCHS = 50

# Update the paths
PARKINSON_PATH = 'C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\classifier\\PD'
NON_PARKINSON_PATH = 'C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\classifier\\ControlSWEDD'

def load_data(parkinson_path, non_parkinson_path):
    images = []
    masks = []
    
    # Load Parkinson's masks
    for mask_path in glob.glob(os.path.join(parkinson_path, '*.png')):  
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMAGE_SIZE)
        images.append(mask)
        masks.append(1)  

  
    for mask_path in glob.glob(os.path.join(non_parkinson_path, '*.png')):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMAGE_SIZE)
        images.append(mask)
        masks.append(0)  
    
    images = np.array(images).astype('float32') / 255.0  
    masks = np.array(masks).astype('float32')  

    return images, masks


images, masks = load_data(PARKINSON_PATH, NON_PARKINSON_PATH)


X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

#Classic segnet arch

def build_segnet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    # Bottleneck
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    
    # Decoder
    up1 = layers.UpSampling2D((2, 2))(conv3)
    conv_up1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up1)

    up2 = layers.UpSampling2D((2, 2))(conv_up1)
    conv_up2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up2)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv_up2)

    model = models.Model(inputs, outputs)
    return model


model = build_segnet((IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

model.save('C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\classifier\\segnet_model.h5')
