try:
    import cv2
    import numpy as np
    import time
    import os
    import sys
    from deepface import DeepFace
    from tkinter import *
except ImportError:
    print("Import error")


def recordFace(frame, count):
    if not os.path.exists("face"):
        os.makedirs("face")
    # save the image
    cv2.imwrite("face/face" + str(count) + ".jpg", frame)
    print("Saved face" + str(count) + ".jpg")


def registerMode(frame):
    if registerMode.count < 5:
        recordFace(frame, registerMode.count)
        registerMode.count += 1
        print(f"Image {registerMode.count} captured.")
        time.sleep(1)  # Delay for feedback
    else:
        print("Image capture limit reached.")


registerMode.count = 0


def compareMode(source_frame):
    # compare face from all images in the face folder
    for file in os.listdir("face"):
        if file.endswith(".jpg"):
            target_frame = cv2.imread("face/" + file)
            result = DeepFace.verify(
                img1_path=source_frame, img2_path=target_frame)
            print(result)
            if result["verified"]:
                print("Face found in " + file)
                return


def main():
    print("Main function")

    # Load the face detection cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(1)  # Changed 1 to 0 for default camera
    print("Camera is open:", camera.isOpened())
    camera.set(3, 640)
    camera.set(4, 480)
    time.sleep(0.1)

    mode = input("Choose mode (register/compare): ").lower()

    if mode == "register":
        print("Register mode selected.")
        mode_function = registerMode
    elif mode == "compare":
        print("Compare mode selected.")
        mode_function = compareMode
    else:
        print("Invalid mode. Exiting.")
        return

    while True:
        ret, frame = camera.read()

        if mode_function is not None:
            mode_function(frame)

        cv2.imshow('Camera Feed', frame)  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
