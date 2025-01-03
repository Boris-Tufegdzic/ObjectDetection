# MediaPartObjectDetection

Real-Time Object Detection using MediaPipe and OpenCV.

## Overview

This project leverages **MediaPipe** and **OpenCV** to perform real-time object detection using a webcam. It uses a pre-trained model (`ssd_mobilenet_v2.tflite`) for detecting objects and displays bounding boxes with category labels and probabilities.


## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python
- A working webcam for real-time detection

### Installation

1. Clone the repository:
   ```bash
   git clone 
   cd MediaPartObjectDetection
    ```

2. (Optionnal) Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. (Optionnal) Activate the virtual environment:

    On Windows:

    ```bash
    venv\Scripts\activate
    ```

    On macOS/Linux:

    ```bash
    source venv/bin/activate
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Ensure the model file (ssd_mobilenet_v2.tflite) is located in the models/ directory.
You can download this model (or others) from here : https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector

## Running the Application

Run the main script:

```bash
cd src
python main.py
```

Press q to exit the detection loop.

