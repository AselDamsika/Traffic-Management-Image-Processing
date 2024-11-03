import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import the video footage
video_path = './src/assets/data_8.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

# Step 2: Ask the user which frame they want to use as the reference for subtraction
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")
frame_to_use = input(f"Enter the frame number to use for subtraction (0 to {total_frames-1}, or press Enter to use the first frame): ")

# Step 3: Use the selected frame or default to the first frame
if frame_to_use.isdigit():
    frame_to_use = int(frame_to_use)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_use)
else:
    frame_to_use = 0  # Default to the first frame

# Step 4: Get the selected frame
ret, reference_frame = cap.read()
if not ret:
    print("Error: Could not read the selected frame.")
    exit()

# Step 5: Convert the reference frame to grayscale
reference_frame_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
reference_frame_gray = cv2.equalizeHist(reference_frame_gray)

# Step 6: Create empty lists for particle speed and paths
particle_speeds = []
particle_paths = []

# Step 7: Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equlized_frame = cv2.equalizeHist(gray_frame)
    
    # Calculate absolute difference between the current frame and the reference frame
    frame_diff = cv2.absdiff(equlized_frame, reference_frame_gray)
    
    # Threshold the difference image to obtain a binary image
    _, binary_frame = cv2.threshold(frame_diff, 125, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((11, 11), np.uint8)
    binary_frame_af_morpho = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
    
    # Track the paths of the moving particles
    contours, _ = cv2.findContours(binary_frame_af_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            
            # Append centroid position to track particle movement
            particle_speeds.append((cx, cy))
            particle_paths.append((cx, cy))
    
    # Draw paths of particles
    for i in range(1, len(particle_paths)):
        if (0 <= particle_paths[i - 1][0] < frame.shape[1] and
                0 <= particle_paths[i - 1][1] < frame.shape[0] and
                0 <= particle_paths[i][0] < frame.shape[1] and
                0 <= particle_paths[i][1] < frame.shape[0]):
            cv2.line(frame, particle_paths[i - 1], particle_paths[i], (0, 0, 255), 2)
    
    # Display the binary frame with paths
    cv2.imshow('Frame', binary_frame_af_morpho)
    
    delay = int(1000 / fps)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Calculate particle speeds based on movement between frames
particle_speeds = np.array(particle_speeds)
if len(particle_speeds) > 1:
    distances = np.linalg.norm(particle_speeds[1:] - particle_speeds[:-1], axis=1)
    speeds = distances * fps  # Speed = distance * frame rate

    # Plot speeds over time
    plt.plot(speeds)
    plt.xlabe
