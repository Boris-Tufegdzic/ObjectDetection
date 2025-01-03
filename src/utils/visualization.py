import cv2

def visualize(image, detection_result):
    """Function to visualize detection results : Draws bounding boxes and labels on the image."""
    for detection in detection_result.detections:
        # Draw bounding box
        bbox = detection.bounding_box
        start_point = int(bbox.origin_x), int(bbox.origin_y) #top left corner
        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height) #bottom right corner
        cv2.rectangle(image, start_point, end_point, (255, 0, 0), 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (int(bbox.origin_x) + 10, int(bbox.origin_y) - 10)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return image
