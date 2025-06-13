# FastRTC YOLOv10 Object Detection App

This application uses FastRTC and YOLOv10 to perform real-time object detection on a webcam feed or mobile camera.

## Features

- Real-time object detection using YOLOv10n.
- Web interface powered by Gradio and FastRTC.
- Adjustable confidence threshold for detections.

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Run the Gradio application:**
    ```bash
    python app.py
    ```
2.  Open your web browser and navigate to the URL provided by Gradio (usually `http://127.0.0.1:7860` or a similar local address).
3.  Allow camera access if prompted by your browser.

## How it Works

-   `app.py`: Contains the main Gradio application logic, sets up the FastRTC stream, and handles video processing.
-   `inference.py`: Implements the `YOLOv10` class, which loads the ONNX model, preprocesses images, runs inference, and draws detections.
-   `requirements.txt`: Lists the necessary Python packages.
-   The YOLOv10n ONNX model is downloaded automatically from Hugging Face Hub upon first run.
