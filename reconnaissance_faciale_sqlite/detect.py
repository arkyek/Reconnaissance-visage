# -*- coding: cp1252 -*-

#importer les 2 libs 'numpy' et openCv
import cv2
import numpy as np
import sqlite3

recognizer = cv2.createLBPHFaceRecognizer()  #cv2.face.createFisherFaceRecognizer()	#createEigenFaceRecognizer()
recognizer.load('trainner/trainner.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#fonction qui nous permet d'obtenir le profile de la bd
def getProfile(Id):
	conn=sqlite3.connect("facesDB")
	cmd="SELECT * FROM personnes WHERE ID="+str(Id)
	cursor=conn.execute(cmd)
	profile=None
	for row in cursor:
		profile=row
	conn.close()
	return profile

Id=0
video_capture = cv2.VideoCapture(0)
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1,1,0,1)

while True:
        ret, img = video_capture.read()
        
        if ret: 
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        
                for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

				#prévoir la personne détécté a partir de notre modèle LBPH
        		Id,conf=recognizer.predict(gray[y:y+h, x:x+w])

			#appele à la fonction getProfile afin d'obtenir les infos de la bd
			profile=getProfile(Id)
			#si l'id existe on écrit à coté du rectangle qui entour les visages détéctés les informations enregistrée dans la bd
			if(profile!=None):
				cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[1]),(x,y+h+15),font,255)
				cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[2]),(x,y+h+32),font,255)
				cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[3]),(x,y+h+47),font,255)
				cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[4]),(x,y+h+62),font,255)
			else:
				cv2.cv.PutText(cv2.cv.fromarray(img),str("Inconnue"),(x,y+h+15),font,255)
                cv2.imshow('Video', img)
        
        #ECHAP pour quitter
        k = cv2.waitKey(30) & 0xff
        if k == 27:
                break

video_capture.release()
cv2.destroyAllWindows()
