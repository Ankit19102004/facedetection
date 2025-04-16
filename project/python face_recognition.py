import cv2
import face_recognition
import os
from PIL import Image

# Function to ensure image is RGB and save as a processed version
def validate_and_convert_images(image_paths):
    processed_image_paths = []
    for image_path in image_paths:
        try:
            # Open the image and ensure it's RGB
            with Image.open(image_path) as img:
                print(f"Processing: {image_path}")
                if img.mode != "RGB":
                    print(f"Converting to RGB: {image_path}")
                    img = img.convert("RGB")
                # Save as a new file
                output_path = image_path.replace(".", "_processed.")
                img.save(output_path, "JPEG")
                print(f"Saved processed image: {output_path}")
                processed_image_paths.append(output_path)
            processed_image_paths.append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    return processed_image_paths

# Function to load and encode image
def load_and_encode_image(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            return encodings[0]
        else:
            print(f"No face detected in {file_path}. Skipping.")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Image paths and names
image_paths = ["modiji.jpg"
 #  "ankit.jpeg",
#   "mess.jpg",
#     "ronaldo.jpg",
#     "kingkohli.jpg"
]
names = ["ANKIT", "MESSI", "RONALDO", "KOHLI"]

# Convert images to RGB and ensure proper format
processed_image_paths = validate_and_convert_images(image_paths)

# Load and encode images
known_face_encodings = []
known_face_names = []

for path, name in zip(processed_image_paths, names):
    encoding = load_and_encode_image(path)
    if encoding is not None:
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# Verify encodings
if not known_face_encodings:
    print("No valid encodings found. Exiting.")
    exit()

# Video capture and face recognition
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Process each face found
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = None
        if len(face_distances) > 0:
            best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale back face locations to the original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle and label around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the video feed
    cv2.imshow("Video", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
