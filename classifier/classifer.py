import cv2
import numpy as np
import tensorflow as tf

# Load  model
model = tf.keras.models.load_model('C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\classifier\\segnet_model.h5')


def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    # Expand dimensions to match model input shape
    image = np.expand_dims(image, axis=-1)  
    image = np.expand_dims(image, axis=0)   
    return image

def classify_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    
    if prediction[0][0] > 0.5:
        return "Parkinson's"
    else:
        return "Non-Parkinson's"

image_path = 'C:\\Users\\Utsav\\OneDrive\\Desktop\\research\\diy\\classifier\\test.jpg'  
result = classify_image(image_path)
print(f"The image is classified as: {result}")
