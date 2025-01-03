import cv2
import mediapipe as mp
from utils.detector import create_detector
from utils.camera_utils import initialize_camera, read_frame
from utils.visualization import visualize

def main():
    # Initialize MediaPipe Object Detector
    detector = create_detector()

    # Initialize webcam
    cap = initialize_camera()

    print("Press 'q' to exit.")

    while True:
        frame = read_frame(cap)
        if frame is None:
            break

        # Convert frame to an MP Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Object detection
        detection_result = detector.detect(mp_frame)

        # Visualization
        annotated_frame = visualize(frame, detection_result)

        # Display the frame
        cv2.imshow("Real-Time Object Detection", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
