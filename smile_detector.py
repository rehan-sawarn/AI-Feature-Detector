import cv2
#Face Classifier
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
#Smile Classifier
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')

#Grab Webcam feed
webcam=cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    #Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)

    #Run smile detection within each of those faces
    for (x,y,w,h) in faces:
        #draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200,50), 4)
    
    #get the sub frame 
    the_face = frame[y:y+h, x:x+w] 

    #Change just the face to grayscale
    face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
    
    #Detect smiles in the face
    smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

    for (x,y,w,h) in smiles:
        #draw a rectangle around the face
        cv2.rectangle(the_face, (x,y), (x+w, y+h), (50,50,200), 4)    

    #show the image
    cv2.imshow('Smile Detector', frame)
    cv2.waitKey(1)

#cleanup
webcam.release()
cv2.destroyAllWindows()        