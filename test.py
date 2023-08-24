import cv2
import os
import face_recognition


def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            person_image_filenames = [
                filename for filename in os.listdir(person_dir)
                if filename.endswith(".jpg") or filename.endswith(".png")
            ]
            for filename in person_image_filenames:
                print(f"Loading {person_name}/{filename}")
                image_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)
                print(f"Loaded {person_name}/{filename}")

    return known_face_encodings, known_face_names


def add_new_person(camera, known_faces_dir, known_face_names):
    while True:
        name = input("Enter the name of the new person: ")
        if name in known_face_names:
            print("This name already exists. Please choose a different name.")
        else:
            break

    person_dir = os.path.join(known_faces_dir, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    num_photos = 0

    while num_photos < 5:
        ret, frame = camera.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                if w > 200 and h > 200:  # Check if face size is not too small
                    face = frame[y:y+h, x:x+w]
                    if face.shape[0] > 0 and face.shape[1] > 0:
                        cv2.imshow('Capture Face', face)
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            face_path = os.path.join(
                                person_dir, f"{name}_{num_photos}.jpg")
                            cv2.imwrite(face_path, face)
                            num_photos += 1
                            print(f"Saved {num_photos} photo(s) for {name}")

    cv2.destroyAllWindows()


def recognize_face(frame, known_face_encodings, known_face_names):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        # Convert to RGB format
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(face_rgb)

        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = matches.index(True) if True in matches else -1
            name = known_face_names[best_match_index] if best_match_index != - \
                1 else "Unknown"
            accuracy = 1 - \
                face_distances[best_match_index] if best_match_index != -1 else 0
            if accuracy < 0.6:
                name = "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({accuracy:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


def main():
    camera = cv2.VideoCapture(0)  # Change to 0 for default camera

    known_faces_dir = "face"
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    while True:
        print("Select an option:")
        print("1. Recognize Faces")
        print("2. Add New Person")
        print("3. Quit")
        choice = input()

        if choice == '1':
            while True:
                ret, frame = camera.read()
                if ret:
                    processed_frame = recognize_face(
                        frame, known_face_encodings, known_face_names)
                    cv2.imshow('frame', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        elif choice == '2':
            add_new_person(camera, known_faces_dir, known_face_names)
        elif choice == '3':
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
