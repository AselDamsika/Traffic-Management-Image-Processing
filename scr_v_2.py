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
line_orientation = "horizontal"  # Choose between 'horizontal' or 'vertical'
line_position = 400  # For horizontal: y-coordinate, For vertical: x-coordinate

# Dictionary to store centroids of detected vehicles
centroid_dict = {}

# Step 4: Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Optional: Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480))  # Adjust resolution if needed

    # Step 5: Apply background subtraction to detect moving objects
    fg_mask = background_subtractor.apply(frame)

    # Step 6: Morphological operations to remove noise and fill gaps
    kernel = np.ones((9, 9), np.uint8)
    binary_frame_af_morpho = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Step 7: Find contours of moving objects
    contours, _ = cv2.findContours(binary_frame_af_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 300  # Adjust this based on the size of vehicles
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Get the bounding box around the vehicle
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the centroid of the vehicle
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            # Add the centroid to the dictionary with a unique ID for each vehicle
            centroid_dict[frame_count] = (cx, cy)

            # Check if the vehicle is crossing the virtual line based on orientation
            if line_orientation == "horizontal" and cy > line_position:  # Horizontal line
                vehicle_count += 1
                del centroid_dict[frame_count]  # Avoid double counting
            elif line_orientation == "vertical" and cx > line_position:  # Vertical line
                vehicle_count += 1
                del centroid_dict[frame_count]  # Avoid double counting

            # Optionally, draw bounding boxes and centroid for visualization
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Mark the centroid

    # Step 8: Draw the virtual counting line based on orientation
    if line_orientation == "horizontal":
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)
    elif line_orientation == "vertical":
        cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (255, 0, 0), 2)

    # Step 9: Display the frame with the counting line and bounding boxes (optional)
    cv2.imshow('Frame', binary_frame_af_morpho)

    delay = int(1000 / fps)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Step 10: Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Step 11: Print the total number of detected vehicles that crossed the line
print(f"Total number of vehicles detected: {vehicle_count}")
