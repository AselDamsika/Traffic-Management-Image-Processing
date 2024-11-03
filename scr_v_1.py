import cv2
import numpy as np
import os

# Step 1: Import the video footage
cap = cv2.VideoCapture('./src/assets/data_8.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Step 2: Initialize background subtractor for better motion detection
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

# Step 3: Create folder for saving vehicle images
output_folder = './extracted_vehicles'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0
vehicle_count = 0  # Vehicle detection counter

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
    kernel = np.ones((5, 5), np.uint8)
    binary_frame_af_morpho = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Step 7: Find contours of moving objects
    contours, _ = cv2.findContours(binary_frame_af_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 500  # Adjust this based on the size of vehicles
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Get the bounding box around the vehicle
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the vehicle from the frame
            vehicle = frame[y:y+h, x:x+w]

            # Save the extracted vehicle as an image
            vehicle_filename = f"{output_folder}/vehicle_{vehicle_count}.png"
            cv2.imwrite(vehicle_filename, vehicle)
            vehicle_count += 1  # Increment vehicle count

            # Optionally, draw bounding boxes on the frame for visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Step 8: Display the frame with bounding boxes (optional)
    cv2.imshow('Frame', frame)

    delay = int(1000 / fps)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Step 9: Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Step 10: Print the total number of detected vehicles
print(f"Total number of vehicles detected: {vehicle_count}")
