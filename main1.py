import math
import cv2
import cvzone
import torch
import os
from numberplate import predict_number_plate
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load model only once

# Load the YOLO model
model = YOLO("best.pt")  # Make sure to update the path to your trained model

# Set the device for the model (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names (as per your model's classes)
classNames = ["with helmet", "without helmet", "rider", "number plate"]

# Initialize variables
li = dict()
rider_box = list()

# Folder to save extracted number plates
output_folder = "extracted_number_plates"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Path to the text file where the number plates will be saved
text_file_path = "detected_number_plates.txt"

# Open the text file in write mode (it will create the file if it doesn't exist)
with open(text_file_path, "w") as file:

    # Function to process an image and make predictions
    def process_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("Error loading image!")
            return
        
        # Convert image to RGB (YOLO expects RGB format)
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run the YOLO model on the image
        results = model(new_img, stream=True, device=device)
        
        for r in results:
            boxes = r.boxes
            xy = boxes.xyxy  # Get the coordinates of bounding boxes
            confidences = boxes.conf  # Get the confidence values
            classes = boxes.cls  # Get the class ids
            
            new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)

            try:
                new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
                indices = torch.where(new_boxes[:, -1] == 2)  # Class ID 2 is for the "rider"
                rows = new_boxes[indices]
                
                for box in rows:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    rider_box.append((x1, y1, x2, y2))
            except Exception as e:
                print(f"Error processing bounding boxes: {e}")
            
            for i, box in enumerate(new_boxes):
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box[4] * 100)) / 100
                cls = int(box[5])

                if classNames[cls] == "without helmet" and conf >= 0.5 or classNames[cls] == "rider" and conf >= 0.45 or \
                        classNames[cls] == "number plate" and conf >= 0.5:
                    if classNames[cls] == "rider":
                        rider_box.append((x1, y1, x2, y2))
                    if rider_box:
                        for j, rider in enumerate(rider_box):
                            if x1 + 10 >= rider_box[j][0] and y1 + 10 >= rider_box[j][1] and x2 <= rider_box[j][2] and \
                                    y2 <= rider_box[j][3]:
                                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                                cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                                   offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                                li.setdefault(f"rider{j}", [])
                                li[f"rider{j}"].append(classNames[cls])

                                if classNames[cls] == "number plate":
                                    npx, npy, npw, nph, npconf = x1, y1, w, h, conf
                                    crop = img[npy:npy + h, npx:npx + w]

                                if li:
                                    for key, value in li.items():
                                        if key == f"rider{j}":
                                            if len(list(set(li[f"rider{j}"]))) == 3:
                                                try:
                                                    vehicle_number, conf = predict_number_plate(crop, ocr)
                                                    if vehicle_number and conf:
                                                        cvzone.putTextRect(img, f"{vehicle_number} {round(conf*100, 2)}%",
                                                                           (x1, y1 - 50), scale=1.5, offset=10,
                                                                           thickness=2, colorT=(39, 40, 41),
                                                                           colorR=(105, 255, 255))

                                                        # Save the cropped number plate image to the output folder
                                                        number_plate_image_path = os.path.join(output_folder, f"{vehicle_number}_{round(conf*100)}.jpg")
                                                        cv2.imwrite(number_plate_image_path, crop)  # Save the image

                                                        # Write the detected number plate and confidence to the text file
                                                        file.write(f"{vehicle_number} - {round(conf * 100, 2)}%\n")
                                                except Exception as e:
                                                    print(f"Error in number plate prediction: {e}")

        # Display the final image with annotations
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Path to the image to process
image_path = 'Data/images/new40.jpg'  # Replace with the path to your image
process_image(image_path)
