import os

# Required Libraries
required_libraries = [
    "tensorflow",
    "keras",
    "numpy",
    "pandas",
    "opencv-python",
    "matplotlib",
    "paddleocr",
    "ultralytics",  # For YOLOv8 (use 'yolov5' if using YOLOv5)
    "streamlit",
    "flask",  # Use flask or fastapi (choose based on your backend)
    "fastapi",
    "uvicorn",
    "reportlab",
    "pypdf2"
]

# Install missing libraries
for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        os.system(f"pip install {lib}")
