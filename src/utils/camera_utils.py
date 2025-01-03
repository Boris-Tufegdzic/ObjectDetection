import cv2

def initialize_camera(camera_id=0):
    """Initializes the webcam and returns the capture object."""
    cap = cv2.VideoCapture(camera_id) # video stream (0 for default webcam)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()
    return cap

def read_frame(cap):
    """Reads a frame from the webcam."""
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None
    return frame
