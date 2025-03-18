import os
import cv2
import numpy as np
import torch
import math
from paddleocr import PaddleOCR
from ultralytics import YOLO
import streamlit as st
import tempfile
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
from datetime import datetime

# Function to convert a file to base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-position: center;
        background-size: cover;
        font-family: "Times New Roman", serif;
    }}
    h1, h2, h3, p {{
        font-family: "Times New Roman", serif;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image for the app
set_background('Background/6.jfif')

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load model only once

# Load the YOLO model
model = YOLO("best.pt")  # Make sure to update the path to your trained model

# Set the device for the model (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names (as per your model's classes)
classNames = ["with helmet", "without helmet", "rider", "number plate"]

# Folder to save extracted number plates
output_folder = "extracted_number_plates"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Path to the text file where the number plates will be saved
text_file_path = "detected_number_plates.txt"

# Fine amount for not wearing a helmet
helmet_fine = 500  # Fine amount in local currency (e.g., 500)

# Function to read the user_login_name from a text file
def read_user_login_name():
    user_login_file = "logged_in_user.txt"
    if os.path.exists(user_login_file):
        with open(user_login_file, "r") as f:
            return f.read().strip()  # Read and strip any extra whitespace
    return "Admin"  # Default value if the file doesn't exist

# Function to generate challan PDF
def generate_challan_pdf(name, vehicle_id, fine_amount, user_login_name):
    # Create a BytesIO object to store PDF in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Set font
    c.setFont("Helvetica", 12)
    
    # Title of the challan
    c.drawString(200, 750, "TRAFFIC CHALLAN")
    c.line(50, 740, 550, 740)
    
    # Add challan details
    details = [
        ("Challan Number: ", "CHN-1234567"),
        ("Date: ", datetime.now().strftime("%Y-%m-%d")),
        ("Police By: ", user_login_name),  # Add the user's login name
        ("Owner Name: ", name),
        ("Vehicle Number: ", vehicle_id),
        ("Fine for not wearing helmet: ", f"₹{fine_amount}"),
        ("Total Fine: ", f"₹{fine_amount}")
    ]
    
    y_offset = 700
    for label, value in details:
        c.drawString(50, y_offset, f"{label} {value}")
        y_offset -= 20  # Move to next line
    
    # Footer
    footer = "Please pay your fine within 7 days to avoid additional penalties."
    c.drawString(50, y_offset, footer)
    c.line(50, y_offset - 10, 550, y_offset - 10)
    
    # Save PDF to the buffer
    c.showPage()
    c.save()
    
    # Move buffer position to the beginning
    buffer.seek(0)
    
    return buffer, details

# Function to save challan details to Excel
def save_to_excel(details):
    # Convert details to a DataFrame
    df = pd.DataFrame([dict(details)])
    
    # Check if the Excel file already exists
    excel_file = "challan_details.xlsx"
    if os.path.exists(excel_file):
        # Load existing data and append new data
        existing_df = pd.read_excel(excel_file)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df
    
    # Save the updated DataFrame to Excel
    updated_df.to_excel(excel_file, index=False)

# Function to perform OCR and extract license plate number
def extract_license_plate_from_image(cropped_image, crop_filename):
    image_np = np.array(cropped_image)  # Convert to numpy array for PaddleOCR
    result = ocr.ocr(image_np, cls=True)  # Perform OCR with angle classification
    
    detected_license_plate = ""
    for line in result[0]:
        detected_license_plate = line[1][0]  # Extract the detected text (vehicle number)
        print(f"Detected text: {detected_license_plate}")
    
    # Save the license plate text into a file with the same name as the image (but with .txt extension)
    text_file_path = crop_filename.replace(".jpg", ".txt")
    
    # Check if the output folder exists, if not create it
    output_folder = "extracted_texts"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the detected license plate number into the text file
    text_file_path = os.path.join(output_folder, os.path.basename(text_file_path))
    with open(text_file_path, "w") as f:
        f.write(detected_license_plate)
    
    print(f"License plate saved in text file: {text_file_path}")
    
    return detected_license_plate

# Function to save video frames
def save_video_frames(video_file):
    frames_folder = "video_frames"
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    # Save the uploaded video file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_file.read())
        temp_video_path = temp_video_file.name

    # Open the video file using OpenCV
    video = cv2.VideoCapture(temp_video_path)
    
    # Check if the video was opened successfully
    if not video.isOpened():
        st.error("Failed to open video.")
        return

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_filename = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    video.release()

    # Clean up the temporary video file after processing
    os.remove(temp_video_path)
    
    st.success(f"Saved {frame_count} frames to {frames_folder}")

# Function to process an image and make predictions
def process_image(image_bytes, user_login_name):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Error loading image!")
        return
    
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device=device)
    
    fine_amount = 0  # Initialize fine amount
    detected_license_plate = ""  # Initialize license plate
    
    cropped_images_info = []  # List to store information about cropped images

    with open(text_file_path, "w") as file:
        for r in results:
            boxes = r.boxes
            xy = boxes.xyxy
            confidences = boxes.conf
            classes = boxes.cls
            new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)

            try:
                new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
                indices = torch.where(new_boxes[:, -1] == 2)
                rows = new_boxes[indices]
                
                for box in rows:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            except Exception as e:
                print(f"Error processing bounding boxes: {e}")
            
            for i, box in enumerate(new_boxes):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box[4] * 100)) / 100
                cls = int(box[5])

                if classNames[cls] == "without helmet" and conf >= 0.5:
                    fine_amount += helmet_fine

                if classNames[cls] == "without helmet" and conf >= 0.5 or classNames[cls] == "rider" and conf >= 0.45 or classNames[cls] == "number plate" and conf >= 0.5:
                    color = (255, 0, 0)  # Blue for bounding box
                    thickness = 2
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    label = f"{classNames[cls]} {conf:.2f}"
                    img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if classNames[cls] == "without helmet" or classNames[cls] == "rider" or classNames[cls] == "number plate":
                        crop = img[y1:y2, x1:x2]
                        cropped_folder = "cropped_images"
                        if not os.path.exists(cropped_folder):
                            os.makedirs(cropped_folder)
                        
                        crop_filename = os.path.join(cropped_folder, f"{classNames[cls]}_{x1}_{y1}.jpg")
                        cv2.imwrite(crop_filename, crop)

                        if classNames[cls] == "number plate":
                            try:
                                # Use the OCR function to extract the license plate number
                                vehicle_number = extract_license_plate_from_image(crop, crop_filename)
                                if vehicle_number:
                                    detected_license_plate = vehicle_number
                                    cropped_images_info.append({
                                        "image_filename": crop_filename,
                                        "vehicle_number": vehicle_number
                                    })
                                    file.write(f"Vehicle Number: {vehicle_number}, Image: {crop_filename}\n")
                            except Exception as e:
                                print(f"Error extracting license plate from crop: {e}")

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.png', img_bgr)
        img_bytes = img_encoded.tobytes()
        
        st.image(img_bytes, caption="Processed Image with Bounding Boxes", use_column_width=True)
        st.text(f"Detected number plates saved in: {text_file_path}")
        
        if fine_amount > 0:
            st.write(f"Fine for not wearing helmet: ₹{fine_amount}")
        else:
            st.write("No fine detected. All riders are wearing helmets.")
        
        # Input fields for challan details
        name = st.text_input("Enter name", value="John Doe")
        
        # Automatically fill the vehicle number if detected
        if detected_license_plate:
            vehicle_id = st.text_input("Enter license plate number", value=detected_license_plate)
        else:
            vehicle_id = st.text_input("Enter license plate number", value="")
        
        fine_amount_input = st.number_input("Enter fine amount", min_value=0, value=fine_amount)

        # Save challan details
        if st.button("Save Challan"):
            # Save challan details to Excel
            challan_details = [
                ("Challan Number: ", "CHN-1234567"),
                ("Date: ", datetime.now().strftime("%Y-%m-%d")),
                ("Issued By: ", user_login_name),  # Add the user's login name
                ("Owner Name: ", name),
                ("Vehicle Number: ", vehicle_id),
                ("Fine for not wearing helmet: ", f"₹{fine_amount_input}"),
                ("Total Fine: ", f"₹{fine_amount_input}")
            ]
            save_to_excel(challan_details)
            st.success("Challan details saved to Excel.")
            st.session_state.challan_saved = True  # Set a session state flag

        # Display download button only if challan is saved
        if st.session_state.get("challan_saved", False):
            challan_pdf, _ = generate_challan_pdf(name, vehicle_id, fine_amount_input, user_login_name)
            st.download_button(
                label="Download Challan PDF",
                data=challan_pdf,
                file_name="challan.pdf",
                mime="application/pdf"
            )

# Streamlit user interface
def main():
    st.title("Vehicle Number Plate Detection and Challan Generation")
    st.write("Upload an image to detect vehicle number plates, check helmet compliance, and generate a challan.")
    
    # Read the user_login_name from the text file
    user_login_name = read_user_login_name()
    
    # Store the user_login_name in the session state
    if 'user_login_name' not in st.session_state:
        st.session_state.user_login_name = user_login_name
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    uploaded_video = st.file_uploader("Or upload a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        process_image(image_bytes, user_login_name)

    if uploaded_video is not None:
        save_video_frames(uploaded_video)

if __name__ == "__main__":
    main()