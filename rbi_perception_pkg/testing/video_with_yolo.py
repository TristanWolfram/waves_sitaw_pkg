import cv2
from ultralytics import YOLO

video_name = "sample1"
input_path = f'rbi_perception_pkg/testing/videos/{video_name}.mp4'
output_path = f'rbi_perception_pkg/testing/videos/{video_name}_with_detections.mp4'
# Path to the Ultralytics YOLO model file (e.g., 'yolov8s.pt')
model_path = 'yolo12n.pt'
# Confidence threshold for detections
conf_thres = 0.25
# Desired output frames per second (playback speed control)
output_fps = 30  # e.g., 30 FPS for faster playback


def main():
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Original FPS (unused for output speed control)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input video FPS: {input_fps}")

    # Prepare output video writer with specified output_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference with confidence threshold
        results = model(frame, conf=conf_thres)
        # Annotate frame (draw boxes, labels)
        annotated_frame = results[0].plot()

        # Write annotated frame to output
        out.write(annotated_frame)

        # Optional display (press 'q' to quit early)
        cv2.imshow('YOLO Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
