

import streamlit as st
import pandas as pd
import hashlib
import os
import openpyxl
import cv2
import pytesseract
import base64
import subprocess
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

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

# File for storing officer data (create if not exists)
officer_data_file = "officers.xlsx"

# File for storing challan details (create if not exists)
challan_details_file = "challan_details.xlsx"

# Initialize or load officer data from Excel
def load_officer_data():
    if os.path.exists(officer_data_file):
        return pd.read_excel(officer_data_file)
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=["police_name", "username", "police_id", "station", "branch", "state", "password"])
        df.to_excel(officer_data_file, index=False)
        return df

# Save officer data to Excel
def save_officer_data(df):
    df.to_excel(officer_data_file, index=False)

# Initialize or load challan details from Excel
def load_challan_data():
    if os.path.exists(challan_details_file):
        return pd.read_excel(challan_details_file)
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=["Challan Number", "Date", "Owner Name", "Vehicle Number", "Fine Amount", "Issued By"])
        df.to_excel(challan_details_file, index=False)
        return df

# Save challan details to Excel
def save_challan_data(df):
    df.to_excel(challan_details_file, index=False)

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Predefined Admin credentials
admin_username = "Admin"
admin_password = "admin123"

# Register Officer
def register():
    st.title("Police Officer Registration")
    st.write("Please fill in the details below to register as a police officer.")
    
    with st.form("registration_form"):
        police_name = st.text_input("Police Name")
        username = st.text_input("Username")
        police_id = st.text_input("Police ID")
        station = st.text_input("Station")
        branch = st.text_input("Branch")
        state = st.text_input("State")
        password = st.text_input("Password", type='password')
        confirm_password = st.text_input("Confirm Password", type='password')
        
        if st.form_submit_button("Register"):
            if password == confirm_password:
                # Load officer data from Excel
                officer_df = load_officer_data()

                # Check if the username already exists
                if username in officer_df['username'].values:
                    st.error("Username already exists! Please choose a different one.")
                else:
                    # Create a new DataFrame for the new officer
                    new_data = pd.DataFrame([{
                        "police_name": police_name,
                        "username": username,
                        "police_id": police_id,
                        "station": station,
                        "branch": branch,
                        "state": state,
                        "password": hash_password(password)
                    }])

                    # Concatenate the new data with the existing DataFrame
                    officer_df = pd.concat([officer_df, new_data], ignore_index=True)

                    # Save updated officer data back to Excel
                    save_officer_data(officer_df)
                    st.success("Registration successful! Please log in.")
            else:
                st.error("Passwords do not match!")

# Login function
def login(is_admin=False):
    st.title("Admin Login" if is_admin else "User Login")
    st.write("Please enter your credentials to log in.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        
        if st.form_submit_button("Login"):
            if is_admin:
                if username == admin_username and password == admin_password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.is_admin = True
                    st.success(f"Login successful! Welcome Admin.")
                    
                    # Save the username to a text file
                    with open("logged_in_user.txt", "w") as f:
                        f.write(username)
                    
                    admin_check()
                else:
                    st.error("Invalid Admin credentials!")
            else:
                # Load officer data from Excel
                officer_df = load_officer_data()

                # Search for the officer in the DataFrame
                officer = officer_df[officer_df['username'] == username]

                if not officer.empty:
                    # Check password match after hashing both
                    stored_password_hash = officer['password'].values[0]
                    entered_password_hash = hash_password(password)

                    if stored_password_hash == entered_password_hash:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.is_admin = is_admin
                        st.success(f"Login successful! Welcome {username}.")
                        
                        # Save the username to a text file
                        with open("logged_in_user.txt", "w") as f:
                            f.write(username)
                        
                        subprocess.run(["streamlit", "run", "app1.py"])
                    else:
                        st.error("Invalid credentials! Please check your username and password.")
                else:
                    st.error("Username not found!")

# Admin Check Functionality
def admin_check():
    st.title("Admin Check - Challan Details")
    
    # Load challan data from Excel
    challan_df = load_challan_data()
    
    if not challan_df.empty:
        st.write("### Challan Details")
        
        # Display all challans
        st.dataframe(challan_df)
        
        # Display the count of challans created by each officer
        challans_by_user = challan_df['Issued By: '].value_counts()
        
        st.write("### Number of Challans by Each Officer")
        for user, count in challans_by_user.items():
            st.write(f"{user}: {count} challans")
        
    else:
        st.write("No challan details found.")

# Home Page
def home_page():
    st.title("Welcome to the Challan Generation System")
    st.write("Please select an option from the sidebar to proceed.")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    action = st.sidebar.radio("Select Action", ["Home", "Admin Login", "User Login", "Register"])

    if action == "Home":
        st.write("This is the home page. Please select an option from the sidebar.")
    elif action == "Admin Login":
        login(is_admin=True)
    elif action == "User Login":
        login(is_admin=False)
    elif action == "Register":
        register()

# Main function to route between pages
def main():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        home_page()
    else:
        if st.session_state.get("is_admin", False):
            admin_check()
        else:
            st.write("Welcome to the dashboard!")

# Run the application
if __name__ == "__main__":
    main()
