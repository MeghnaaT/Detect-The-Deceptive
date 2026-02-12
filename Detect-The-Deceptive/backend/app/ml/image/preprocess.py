import cv2
import numpy as np

def preprocess_image(upload_file, return_original=False):
    image_bytes = upload_file.file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)

    original = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.resize(original, (224, 224))
    img = img.astype("float32") / 255.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    if return_original:
        return img, original

    return img

 # adds a new dimension at the front
                                       # if img has a shape(244, 244,3) , after this step it becomes (1, 244, 244,3)
# 1) Read raw bytes from file.
# 2) Convert bytes → NumPy array.
# 3) Decode into an image (BGR format).
# 4) Resize to 224×224.
# 5) Normalize pixel values to [0,1].
# 6) Add batch dimension → ready for model input.
                                   
# Example
# If you upload a JPEG image:
# 1) Original shape: (1024, 768, 3)
# 2) After resize: (224, 224, 3)
# 3) After normalization: pixel values are floats between 0 and 1.
# 4) After expand_dims: (1, 224, 224, 3) → suitable for feeding into a neural network.
