from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load model only once

# Path to the cropped image
image_path = "cropped_images/without helmet_77_581.jpg"

# Use PIL to open the image
image = Image.open(image_path)

# Convert the PIL image to a NumPy array
image_np = np.array(image)

# Perform OCR on the NumPy array image
result = ocr.ocr(image_np, cls=True)

# Print the OCR result
for line in result[0]:
    # line[1] contains the text and confidence
    detected_license_plate = line[1][0]  # The detected text
    confidence = line[1][1]  # The confidence score
    print(f"Detected text: {detected_license_plate}")