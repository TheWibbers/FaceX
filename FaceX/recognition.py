import face_recognition
import os
import cv2
import numpy as np
import math
import time


class FaceRecognition:
    def __init__(self):
        self.known_faces = []  # List to store known faces
        self.authorized_faces = []  # List to store authorized faces
        self.process_current_frame = False
        self.authorize_mode = False
        self.authorize_end_time = None
        self.unknown_counter = 1

    def encode_faces(self, known_faces_dir, authorized_faces_dir):
        self.known_faces = self.load_faces_from_dir(known_faces_dir)
        self.authorized_faces = self.load_faces_from_dir(authorized_faces_dir)

    def load_faces_from_dir(self, dir_path):
        encodings = []
        names = []

        image_files = [f for f in os.listdir(dir_path) if not f.startswith('.') and (f.endswith('.jpg') or f.endswith('.jpeg'))]

        for image_name in image_files:
            image_path = os.path.join(dir_path, image_name)
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            name = os.path.splitext(image_name)[0]

            encodings.append(face_encoding)
            names.append((name, "???"))  # Confidence will be filled in during recognition

        return list(zip(names, encodings))

    def recognize_faces(self, frame, faces_dict):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([item[1] for item in faces_dict], face_encoding)
            name = "Unknown"
            confidence = '???'

            face_distances = face_recognition.face_distance([item[1] for item in faces_dict], face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_dict[best_match_index][0]
                confidence = self.calculate_confidence(face_distances[best_match_index])

            face_names.append((name, confidence))

        return face_locations, face_names


    @staticmethod
    def calculate_confidence(face_distance, face_match_threshold=0.6):
        range_val = 1.0 - face_match_threshold
        linear_val = (1.0 - face_distance) / (range_val * 2.0)

        if face_distance > face_match_threshold:
            confidence = round(linear_val * 100, 2)
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            confidence = round(value, 2)

        return confidence

    def draw_dots(self, frame, locations):
        for (top, right, bottom, left) in locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Calculate facial landmarks
            face_landmarks = face_recognition.face_landmarks(frame, [(top, right, bottom, left)])[0]

            # Draw white dots around the face
            for landmark in face_landmarks.values():
                for point in landmark:
                    cv2.circle(frame, point, 1, (255, 255, 255), -1)

            # Draw Y shape connection of white dots on the eyes and down to the mouth
            eye_points = face_landmarks["left_eye"] + face_landmarks["right_eye"]
            mouth_points = face_landmarks["top_lip"] + face_landmarks["bottom_lip"]
            for i in range(len(eye_points) - 1):
                cv2.line(frame, eye_points[i], eye_points[i + 1], (255, 255, 255), 1)
            for i in range(len(mouth_points) - 1):
                cv2.line(frame, mouth_points[i], mouth_points[i + 1], (255, 255, 255), 1)

    def run_recognition(self, video_capture):
        recognized_faces = set()  # Keep track of recognized faces
        no_face_count = 0

        while True:
            ret, frame = video_capture.read()

            face_locations = []
            face_names = []
            if self.process_current_frame:
                face_locations, face_names = self.recognize_faces(frame, self.known_faces)

                # Update authorization mode
                if any(name != 'Unknown' for name, _ in face_names):
                    self.authorize_mode = True
                    self.authorize_end_time = time.time() + 30  # 30 seconds from now
                elif self.authorize_end_time and time.time() >= self.authorize_end_time:
                    self.authorize_mode = False
                    self.authorize_end_time = None

                # Check if the recognized face meets a certain confidence threshold and hasn't been recognized before
                for name, confidence in face_names:
                    if confidence != '???' and float(confidence) >= 60 and name not in recognized_faces:
                        if name in [item[0] for item in self.authorized_faces]:
                            print("Hello. " + name + ".")  # Print "Hello. [name]." for authorized person
                        recognized_faces.add(name)  # Add the name to the set of recognized faces

                # Resume normal recognition if no unknown faces are detected
                if len(face_names) == 0:
                    recognized_faces = set()

                if len(face_locations) == 0:
                    no_face_count += 1
                else:
                    no_face_count = 0

                if no_face_count >= 20:
                    recognized_faces = set()  # Reset recognized faces
                    no_face_count = 0

                if len(face_names) > 0 and "Unknown" in [name[0] for name in face_names]:
                    unknown_index = [name[0] for name in face_names].index("Unknown")
                    confidence = face_names[unknown_index][1]
                    if confidence != '???':
                        if float(confidence) >= 60:
                            top, right, bottom, left = face_locations[unknown_index]
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            face_image = frame[top:bottom, left:right]
                            cv2.imwrite(f"faces/unknown.jpeg", face_image)

            self.process_current_frame = not self.process_current_frame

            if len(face_locations) > 0:
                self.draw_dots(frame, face_locations)

                for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, f"{name} ({confidence}%)", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Break from loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Close video capture
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_recog = FaceRecognition()
    face_recog.encode_faces('faces/known', 'faces/authorized')

    video_capture = cv2.VideoCapture(0)  # 0 for built-in webcam, 1 for external webcam
    face_recog.run_recognition(video_capture)
