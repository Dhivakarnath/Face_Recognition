import cv2
import face_recognition
import pandas as pd
import numpy as np
import os 

def load_dataset(dataset_dir):
    face_encodings = []
    face_names = []

    # Iterate over each image file in the dataset directory
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(dataset_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]  # Assuming one face per image
            face_encodings.append(encoding)
            face_names.append(os.path.splitext(filename)[0])  # Use filename as the name

    return face_encodings, face_names

def recognize_faces_in_video(face_encodings, face_names, output_csv):
    video_capture = cv2.VideoCapture(0)  # Webcam capture
    name_saved = False  # Flag to track if name has been saved to CSV

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the current face encoding with the known dataset
            matches = face_recognition.compare_faces(face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches and not name_saved:  # Check if match found and name not yet saved
                match_index = np.argmax(matches)
                name = face_names[match_index]

                # Save the recognized name to CSV
                with open(output_csv, 'a') as f:
                    f.write(f"{name}\n")
                
                name_saved = True  # Set flag to True after saving name

                # Display the recognized name on the frame
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if name_saved:  # Exit loop if name has been saved
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_directory = r"C:\Users\dhivu\VS Codes\Face Recognition\data set\Dataset"
    output_csv_file = r"C:\Users\dhivu\VS Codes\Face Recognition\Output\Attendance.csv.xlsx"

    # Load dataset
    face_encodings, face_names = load_dataset(dataset_directory)

    # Perform real-time face recognition using webcam
    recognize_faces_in_video(face_encodings, face_names, output_csv_file)
