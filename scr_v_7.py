import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import the video footage
cap = cv2.VideoCapture('./src/assets/data_8.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
# Load background image as first frame
first_frame = cv2.imread(r'src\assets\bg.png')

# Step 3: Convert the first frame to grayscale
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame_gray = cv2.equalizeHist(first_frame_gray)

# Neighborhood distance threshold (you can adjust this value)
distance_threshold = 100

# Step 4: Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Step 5: Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.medianBlur(gray_frame, ksize=5)
    equlized_frame = cv2.equalizeHist(gray_frame)

    # Step 6: Calculate absolute difference between current frame and first frame
    frame_diff = cv2.absdiff(equlized_frame, first_frame_gray)
    frame_diff = cv2.medianBlur(frame_diff, ksize=21)
    
    # Step 7: Threshold the difference image to obtain a binary image
    _, binary_frame = cv2.threshold(frame_diff, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary_frame_af_morpho = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    
    # Step 8: Find contours
    contours, _ = cv2.findContours(binary_frame_af_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store the bounding boxes of each contour
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # Group nearby bounding boxes
    groups = []
    
    for rect in bounding_boxes:
        x, y, w, h = rect
        new_group = True
        # Check each group
        for group in groups:
            for (gx, gy, gw, gh) in group:
                # If bounding box is close to an existing group, add it to that group
                if abs(gx - x) < distance_threshold and abs(gy - y) < distance_threshold:
                    group.append(rect)
                    new_group = False
                    break
            if not new_group:
                break
        if new_group:
            groups.append([rect])

    # Calculate the centroid of each group
    for group in groups:
        total_x = total_y = total_w = total_h = 0
        for (x, y, w, h) in group:
            total_x += x
            total_y += y
            total_w += w
            total_h += h
        
        # Calculate the center of the group of bounding boxes
        cx = int((total_x + total_w / 2) / len(group))
        cy = int((total_y + total_h / 2) / len(group))

        # Draw the centroid on the original frame
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Display the result
    cv2.imshow('Frame', frame)
    
    delay = int(1000 / fps)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
