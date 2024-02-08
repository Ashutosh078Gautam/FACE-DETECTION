
import tkinter as tk
from tkinter import *
from tkinter import messagebox as ms
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd

from flask import Flask

app = Flask(__name__)



df = pd.read_csv('Profile.csv')
df.sort_values('Ids', inplace = True)
df.drop_duplicates(subset = 'Ids', keep = 'first', inplace = True)
df.to_csv('Profile.csv', index = False)



@app.route('/', methods=["GET"])
def DetectFace():
    reader = csv.DictReader(open('Profile.csv'))
    print('Detecting Login Face')

    # Initialize names to avoid UnboundLocalError
    name1 = ''
    name2 = ''

    for rows in reader:
        result = dict(rows)
        if result['Ids'] == '1':
            name1 = result['Name']
        elif result['Ids'] == '2':
            name2 = result["Name"]

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Use full file path to Trainner.yml
    file_path = os.path.join(os.getcwd(), "TrainData", "Trainner.yml")
    print("File Path:", file_path)


    try:
        recognizer.read(file_path)
        print("Model loaded successfully.")
    except cv2.error as e:
        print(f"Error: {e}")
        return 'Error loading face recognition model'

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    Face_Id = ''

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        Face_Id = 'Not detected'

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            
            if confidence < 80:
                if Id == 1:
                    Predicted_name = name1
                elif Id == 2:
                    Predicted_name = name2
                else:
                    Predicted_name = 'Unknown'

                Face_Id = Predicted_name
            else:
                Face_Id = 'Unknown'

                noOfFile = len(os.listdir("UnknownFaces")) + 1
                if int(noOfFile) < 100:
                    cv2.imwrite("UnknownFaces\\Image" + str(noOfFile) + ".jpg", frame[y:y + h, x:x + w])

        cv2.putText(frame, str(Face_Id), (50, 50), font, 1, (255, 255, 255), 2)
        cv2.imshow('Picture', frame)
        cv2.waitKey(1)

        if Face_Id != 'Not detected':
            print(f'Login successful. User: {Face_Id}')
            return f'User {Face_Id} has been indentified successfully'
        else:
            print('-----------Login failed please try again-------')
            return '!!!~~~~~~|USER NOT FOUND|~~~~~~~~~~!!!'

    # Release the camera and close the OpenCV window
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run()



