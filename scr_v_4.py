import cv2
import numpy as np
import os

# Step 1: Import the video footage
cap = cv2.VideoCapture('./src/assets/data_8.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Step 2: Initialize background subtractor (for vehicle detection)
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

# Step 3: Create folder for saving vehicle images
output_folder = './extracted_vehicles'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Vehicle count and frame count initialization
vehicle_count = 0
frame_count = 0

# Define the orientation and position of the virtual line (user can adjust this)
line_orientation = "vertical"  # Choose between 'horizontal' or 'vertical'
line_position = 400  # For horizontal: y-coordinate, For vertical: x-coordinate

# Dictionary to store previous centroids for persistent tracking
vehicle_dict = {}

# Vehicle ID counter
vehicle_id = 1

# Step 4: Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert the frame to grayscale for thresholding
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(frame_gray, 110, 255, cv2.THRESH_BINARY_INV)

    # Step 6: Morphological operations to remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    binary_frame_af_morpho = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)

    # Step 7: Find contours of moving objects
    contours, _ = cv2.findContours(binary_frame_af_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 8: Remove the largest contour before counting (optional)
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort contours by area
        largest_contour = contours[0]
        if cv2.contourArea(largest_contour) > 1000:  # Remove only if it's very large
            contours = contours[1:]  # Remove the largest contour from the list

    min_contour_area = 500  # Adjust this based on the size of vehicles
    current_vehicles = {}

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Get the bounding box around the vehicle
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the centroid of the vehicle
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            # Check if this vehicle is already being tracked
            new_vehicle = True
            for vid, centroid in vehicle_dict.items():
                # Calculate distance between the new centroid and existing ones
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(centroid['position']))

                # If distance is small, consider this the same vehicle (update its position)
                if dist < 50:  # Distance threshold to avoid duplicate counting
                    vehicle_dict[vid]['position'] = (cx, cy)
                    new_vehicle = False

                    # Check if the vehicle crossed the virtual line
                    if line_orientation == "horizontal":
                        if centroid['position'][1] < line_position and cy >= line_position:
                            vehicle_count += 1
                            print(f"Vehicle {vid} crossed the line (horizontal) at frame {frame_count}.")
                    elif line_orientation == "vertical":
                        if centroid['position'][0] < line_position and cx >= line_position:
                            vehicle_count += 1
                            print(f"Vehicle {vid} crossed the line (vertical) at frame {frame_count}.")
                    break

            # If it's a new vehicle, assign it a new ID and start tracking
            if new_vehicle:
                vehicle_dict[vehicle_id] = {'position': (cx, cy)}
                vehicle_id += 1

            # Optionally, draw bounding boxes and centroid for visualization
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Mark the centroid

    # Step 9: Draw the virtual counting line based on orientation
    if line_orientation == "horizontal":
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)
    elif line_orientation == "vertical":
        cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (255, 0, 0), 2)

    # Step 10: Display the frame with the counting line and bounding boxes (optional)
    cv2.imshow('Frame', frame)

    delay = int(1000 / fps)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Step 11: Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Step 12: Print the total number of detected vehicles that crossed the line
print(f"Total number of vehicles detected: {vehicle_count}")
