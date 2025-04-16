import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load known faces and their names
known_faces = []
known_names = []
face_embeddings = []

# Load and process known face images
def load_known_faces():
    # Dictionary of known faces and their names
    face_images = {
        "ANKIT": "ankit.jpeg",
        "MESSI": "messi1.jpg",
        "RONALDO": "ronaldo.jpg",
        "KOHLI": "kingkohli.jpg"
    }
    
    for name, image_file in face_images.items():
        if os.path.exists(image_file):
            # Read the image
            image = cv2.imread(image_file)
            if image is not None:
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Detect face in the image
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Get the first face found
                    x, y, w, h = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    # Resize face to standard size
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    # Apply histogram equalization for better contrast
                    face_roi = cv2.equalizeHist(face_roi)
                    
                    # Normalize the face image
                    face_roi = face_roi.astype(np.float32) / 255.0
                    
                    # Create face embedding (flatten and normalize)
                    embedding = face_roi.flatten()
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    known_faces.append(face_roi)
                    face_embeddings.append(embedding)
                    known_names.append(name)
                    print(f"Loaded face for {name}")

# Load known faces at startup
load_known_faces()

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Set camera resolution
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def recognize_face(face_img):
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Resize to match known faces
    gray = cv2.resize(gray, (100, 100))
    
    # Normalize the face image
    gray = gray.astype(np.float32) / 255.0
    
    # Create face embedding
    embedding = gray.flatten()
    embedding = embedding / np.linalg.norm(embedding)
    
    # Calculate similarities with all known faces
    similarities = [np.dot(embedding, known_embedding) for known_embedding in face_embeddings]
    
    # Get the best match
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    # If the best match is good enough, return the name
    if best_similarity > 0.6:  # Adjusted threshold for better accuracy
        return known_names[best_match_idx]
    return "Face"

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better face detection
    gray = cv2.equalizeHist(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 50),
        maxSize=(300, 300)
    )
    
    # Process each detected face
    for x, y, w, h in faces:
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Recognize the face
        name = recognize_face(face_roi)
        
        # Draw rectangle and display name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Add face size information
        face_size = f"Size: {w}x{h}"
        cv2.putText(frame, face_size, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()   