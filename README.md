## Face Recognition based login system

Face login using Open CV Python and Flask framework
For facial detection LBPH Recogniser is used.


### Installation and ## Process :-
Install prerequisites dependecies such as:-
`pip install opencv pandas numpy pillow flask`

1. Create some folders : `TrainingImage` & `TrainData`
2. Run Register.py : It will take some input i.e  user id and name needs to be then captures some images for training which is tobe stored in `TrainingImage` folder. After training a `Trainner.yml` file is generated in `TrainData` folder 
3. Run python Login.py to detect the face and Login and login status will be available on browser
4. Default server : (localhost:5000)

