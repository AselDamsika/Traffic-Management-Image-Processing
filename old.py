import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import the video footage
cap = cv2.VideoCapture('./src/assets/data_8.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
# # Step 2: Get the first frame
# ret, first_frame = cap.read()
# if not ret:
#     exit()
first_frame = cv2.imread(r'src\assets\bg.png')
# first_frame = cv2.medianBlur(first_frame,ksize=5)

# Step 3: Convert the first frame to grayscale
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame_gray = cv2.equalizeHist(first_frame_gray)

# Step 4: Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Step 5: Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.medianBlur(gray_frame,ksize=5)
    equlized_frame = cv2.equalizeHist(gray_frame)
    # negative_frame = 255 - gray_frame
    # Step 6: Calculate absolute difference between current frame and first frame
    frame_diff = cv2.absdiff(equlized_frame, first_frame_gray)
    frame_diff = cv2.medianBlur(frame_diff,ksize=21)
    
    # Step 7: Threshold the difference image to obtain a binary image
    _, binary_frame = cv2.threshold(frame_diff, 70, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary_frame_af_morpho = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    
    # Step 8: Track the paths of the moving particles
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Calculate centroid of each particle
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    
    # Display the frame
    cv2.imshow('Frame',binary_frame)
    
    delay = int(1000 / fps)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

