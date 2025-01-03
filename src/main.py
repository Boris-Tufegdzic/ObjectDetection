import cv2
import mediapipe as mp
from utils.detector import create_detector
from utils.camera_utils import initialize_camera, read_frame
from utils.visualization import visualize
import time

def main():
    # Initialize MediaPipe Object Detector
    detector = create_detector()

    # Initialize webcam
    cap = initialize_camera()

    # Variables to calculate FPS
    frame_count = 0
    start_time = None

    print("Press 'q' to exit.")

    while True:
        frame = read_frame(cap)
        if frame is None:
            break

        if start_time is None:
            start_time = time.time()

        # Convert frame to an MP Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Object detection
        detection_start = time.time()
        detection_result = detector.detect(mp_frame)
        detection_end = time.time()

        #FPS for detection
        detection_time = detection_end - detection_start
        fps = 1 / detection_time if detection_time > 0 else 0

        # Visualization
        annotated_frame = visualize(frame, detection_result)

        # Display the frame
        cv2.imshow("Real-Time Object Detection", annotated_frame)

        frame_count += 1
        elapsed_time = time.time() - start_time
        overall_fps = frame_count / elapsed_time

        print(f"Detection FPS: {fps:.2f}, Overall FPS: {overall_fps:.2f}")

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
