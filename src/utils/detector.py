from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

def create_detector(model_path=None, score_threshold=0.5):
    """
    Initializes and returns the MediaPipe object detector.
    Args:
        model_path (str): Path to the model file relative to the script location.
        score_threshold (float): Minimum confidence score for detection.
    """

    models = ['efficientdet_lite0.tflite', 'efficientdet_lite2.tflite', 'ssd_mobilenet_v2.tflite']

    if model_path is None:
        # Construct the absolute path to the model file
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', models[2])
    
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=score_threshold) #object detector configuration
        detector = vision.ObjectDetector.create_from_options(options) #detector initialization
        return detector
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
