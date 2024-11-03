import cv2

# Function to extract a specific frame from the video
def get_frame_from_video(video_path, frame_number):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Set the video to the specified frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Check if frame was successfully read
    if not ret:
        print(f"Error: Could not retrieve frame number {frame_number}.")
        return
    
    # Display the frame (for testing purposes)
    cv2.imshow(f'Frame {frame_number}', frame)
    cv2.imwrite('frame.jpg',frame)
    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release the video capture object
    cap.release()

    return frame

# Example usage
video_path = 'src/assets/data_7.mp4'  # Path to your video file
frame_number = 500  # Frame number to be accessed

# Call the function to retrieve and show the frame
frame = get_frame_from_video(video_path, frame_number)
