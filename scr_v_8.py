import cv2
import numpy as np

# Step 1: Import the video footage
cap = cv2.VideoCapture('./src/assets/data_7.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Step 2: Initialize background subtractor with shadow detection enabled
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=50, detectShadows=False)

# Step 3: Define parameters for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Step 4: Vehicle detection and counting
min_contour_area = 1000  # Lower the minimum contour area to capture smaller vehicles
vehicle_count = 0
frame_count = 0
centroid_dict = {}  # Store centroids for tracking
prev_centroids = []
object_persistence = {}  # Store how long an object has been detected
object_threshold = 10  # Number of frames an object needs to be present for counting

# Virtual line setup for vehicle counting
line_orientation = "vertical"  # Choose between 'horizontal' or 'vertical'
line_position = 500  # Adjust the position to improve vehicle crossing detection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Step 5: Apply background subtraction to detect moving objects
    fg_mask = background_subtractor.apply(frame)

    # Step 6: Remove shadow pixels and other noise
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Step 7: Apply morphological operations to clean the image
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Close gaps

    # Step 8: Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 9: Filter contours based on area and shape (aspect ratio)
    current_centroids = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Filter by aspect ratio (vehicles usually have a ratio between 1.5 and 4)
            if 1.5 < aspect_ratio < 4.0:
                cx, cy = int(x + w / 2), int(y + h / 2)  # Calculate the centroid
                current_centroids.append((cx, cy))

                # Step 10: Count the vehicle if it crosses the virtual line
                # Track the object persistence
                if (cx, cy) in object_persistence:
                    object_persistence[(cx, cy)] += 1
                else:
                    object_persistence[(cx, cy)] = 1

                # Only count the vehicle after it has been detected for a certain number of frames
                if object_persistence[(cx, cy)] > object_threshold:
                    if line_orientation == "vertical" and cx > line_position:
                        vehicle_count += 1
                        # Remove the vehicle from the persistence dictionary to avoid double counting
                        del object_persistence[(cx, cy)]
                    elif line_orientation == "horizontal" and cy > line_position:
                        vehicle_count += 1
                        del object_persistence[(cx, cy)]

                # Optional: Draw bounding box and centroid for visualization
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Step 11: Draw the virtual counting line
    if line_orientation == "horizontal":
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)
    elif line_orientation == "vertical":
        cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (255, 0, 0), 2)

    # Step 12: Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Step 13: Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Step 14: Print the total number of detected vehicles
print(f"Total number of vehicles detected: {vehicle_count}")
