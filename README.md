# 🚦 TRAFFICENFORCER-AI: AI-Based Vehicle Monitoring & Challan Generation System  

## 📌 Project Overview  
Traffic rule violations, especially non-compliance with helmet regulations, pose a significant risk to road safety. Global traffic statistics highlight that a large percentage of two-wheeler accidents involve riders without helmets, leading to severe injuries and fatalities.  

Traditional traffic monitoring methods rely on manual enforcement, where police officers identify violators and issue challans. However, these methods suffer from inefficiencies such as human errors, corruption, delays in fine collection, and high manpower requirements.  

To address these challenges, this project introduces an **AI-based automated vehicle monitoring and challan generation system** that leverages:  
- **YOLO (You Only Look Once)** for real-time helmet detection  
- **PaddleOCR** for license plate recognition  
- **Streamlit-based UI** for user interaction and report generation  
- **Automated PDF generation** for challans and Excel-based record management  

## 🔍 Problem Statement  
The primary issues in the current traffic monitoring system include:  
❌ **Manual Traffic Enforcement**: Human monitoring is inefficient, time-consuming, and inconsistent.  
❌ **Helmet Violation Detection**: Existing e-challan systems lack automated helmet detection.  
❌ **License Plate Recognition Issues**: OCR errors cause incorrect fine issuance.  
❌ **Corruption in Fine Collection**: Manual issuance often leads to bribery and fraudulent practices.  
❌ **Data Management & Record Keeping**: No centralized system for tracking repeated violators.  

## 🎯 Key Features  
✅ **Real-time helmet violation detection** using deep learning  
✅ **Automated license plate recognition** for vehicle identification  
✅ **Challan generation in PDF format**, stored for future reference  
✅ **Efficient law enforcement** by reducing manual effort and corruption  
✅ **Streamlit-based UI** for monitoring and accessing reports  

## 🌍 Social Impact  
This project significantly impacts society in the following ways:  

✔ **Improves Road Safety**: Encourages helmet compliance, reducing accident-related fatalities.  
✔ **Enhances Law Enforcement**: Automated monitoring minimizes human errors and intervention.  
✔ **Reduces Corruption**: Transparent challan generation ensures fair penalty collection.  
✔ **Increases Public Awareness**: Regular enforcement leads to long-term behavioral changes among riders.  

## ⚙️ Tech Stack  
- **Machine Learning**: YOLO for object detection, TensorFlow, Keras  
- **OCR**: PaddleOCR for text extraction from number plates  
- **Backend**: Python (Flask/FastAPI)  
- **Frontend**: Streamlit  
- **Data Processing**: NumPy, Pandas  
- **Data Storage**: Excel/CSV for challan records  
- **PDF Generation**: ReportLab/PyPDF2  

## 🚀 How It Works  
1. The system processes real-time images/videos from traffic cameras.  
2. YOLO detects riders and checks for helmet compliance.  
3. PaddleOCR extracts vehicle registration numbers from detected number plates.  
4. A challan is auto-generated in PDF format and stored in the database.  
5. Law enforcement officials can review reports through the Streamlit dashboard.  

## 📂 Project Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Akshatsachdev/TRAFFICENFORCER-AI.git
   ```
2. Install dependencies:  
   ```bash
   python extensions.py
   ```
3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```
## 📜 License  
This project is open-source and available under the **MIT License**.  

---
