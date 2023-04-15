# Import the necessary packages
import cv2

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

# Create a CascadeClassifier object to detect faces in the video frames
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Enter an infinite loop to continuously capture and process video frames
while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Detect faces in the frame using the cascade classifier
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4)

    # Blur each face found in the frame
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        face_roi = frame[y:y+h, x:x+w]
        
        # Apply a Gaussian blur to the face ROI
        blurred_roi = cv2.GaussianBlur(face_roi, (91, 91), 0)
        
        # Replace the original face ROI with the blurred one
        frame[y:y+h, x:x+w] = blurred_roi
        
    # If no faces were found, display a message on the frame
    if len(faces) == 0:
        cv2.putText(frame, 'No face found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    
    # Display the frame in a window
    cv2.imshow('Face Blur', frame)
    
    # Check for user input to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()