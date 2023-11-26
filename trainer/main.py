import cv2, os
import numpy as np
from PIL import Image
# Path for face image database
path = 'face'
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
# function to get the images and label data
def getImagesAndLabels(path):
    # image path is a folder which contains folders of people's faces and their images in it get the path of all the images in the folder and sub folders and store them in a list imagePaths
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    filters = ['.DS_Store','.text']
    imagePaths = [f for f in imagePaths if not f.endswith(tuple(filters))]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        if os.path.isdir(imagePath):
            imagePaths2 = [os.path.join(imagePath,f) for f in os.listdir(imagePath)]
            imagePaths2 = [f for f in imagePaths2 if not f.endswith(tuple(filters))]
            imagePaths = imagePaths + imagePaths2
    for imagePath in imagePaths:    
        if os.path.isdir(imagePath): continue
        print(imagePath)
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))