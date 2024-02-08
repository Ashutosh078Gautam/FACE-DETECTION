import os
import cv2
import csv
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__, template_folder='templates')


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        id_ = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(image_np)
        ids.append(id_)

    return faces, ids


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        id_ = request.form['id']

        if name.isalpha() and id_.isdigit():
            if int(id_) == 1:
                fieldnames = ['Name', 'Ids']
                with open('Profile.csv', 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow({'Ids': id_, 'Name': name})
            else:
                fieldnames = ['Name', 'Ids']
                with open('Profile.csv', 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({'Ids': id_, 'Name': name})

            return redirect(url_for('take_images', name=name, id_=id_))
        else:
            if name.isalpha():
                error_message = 'Enter Proper Id'
            elif id_.isdigit():
                error_message = 'Enter Proper name'
            else:
                error_message = 'Enter Proper Id and Name'
            return render_template('register.html', error_message=error_message)
    else:
        return render_template('register.html')


@app.route('/take_images/<name>/<id_>')
def take_images(name, id_):
    if not name or not id_:
        return "Invalid request", 400

    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            sampleNum = sampleNum + 1
            cv2.imwrite(f"TrainingImage\\{name}.{id_}.{sampleNum}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('Capturing Face for Login ', img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum > 49:
            break

    cam.release()
    cv2.destroyAllWindows()
    res = f"Images Saved for Name: {name} with ID {id_}"
    print(res)
    print('Images save location is TrainingImage\\')
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True,)