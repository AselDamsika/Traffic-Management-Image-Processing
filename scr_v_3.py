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

# Dictionary to store the last position of centroids to track vehicle movement
previous_centroids = {}

# Step 4: Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert the frame to grayscale for thresholding
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    background_subtractor = cv2.absdiff(frame_gray,bg)

    
    _, binary_frame = cv2.threshold(frame_gray, 110, 255, cv2.THRESH_BINARY_INV)

    # Step 6: Morphological operations to remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    binary_frame_af_morpho = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)

    # Step 7: Find contours of moving objects
    contours, _ = cv2.findContours(binary_frame_af_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 500  # Adjust this based on the size of vehicles
    current_centroids = {}

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > min_contour_area:
            # Get the bounding box around the vehicle
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the centroid of the vehicle
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            # Add the centroid to the dictionary for the current frame
            current_centroids[i] = (cx, cy)

            # Optionally, draw bounding boxes and centroid for visualization
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Mark the centroid

            # Step 8: Check if the vehicle crossed the virtual line
            if i in previous_centroids:
                prev_cx, prev_cy = previous_centroids[i]
                
                if line_orientation == "horizontal":
                    # Check if the vehicle has crossed the horizontal line (top to bottom or bottom to top)
                    if prev_cy < line_position and cy >= line_position:  # Crossing from top to bottom
                        vehicle_count += 1
                        print(f"Vehicle crossed the line (horizontal) at frame {frame_count}.")
                    elif prev_cy > line_position and cy <= line_position:  # Crossing from bottom to top
                        vehicle_count += 1
                        print(f"Vehicle crossed the line (horizontal) at frame {frame_count}.")
                
                elif line_orientation == "vertical":
                    # Check if the vehicle has crossed the vertical line (left to right or right to left)
                    if prev_cx < line_position and cx >= line_position:  # Crossing from left to right
                        vehicle_count += 1
                        print(f"Vehicle crossed the line (vertical) at frame {frame_count}.")
                    elif prev_cx > line_position and cx <= line_position:  # Crossing from right to left
                        vehicle_count += 1
                        print(f"Vehicle crossed the line (vertical) at frame {frame_count}.")

    # Update the previous centroids for the next frame
    previous_centroids = current_centroids.copy()

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
