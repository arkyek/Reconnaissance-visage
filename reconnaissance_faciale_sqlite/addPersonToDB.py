# -*- coding: cp1252 -*-

import cv2
import sqlite3

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

# fonction qui ajoute ou modifie les personnes dans la base de donn�es
def insertOrUpdate(Id,name):
	#connexion a la base de donn�e "faceDB"
	connexion=sqlite3.connect("facesDB")
	#requ�te qui cherche l'id entr� pour verifier s'il existe d�ja dans la BD
	cmd="SELECT * FROM personnes WHERE ID="+str(Id)
	cursor=connexion.execute(cmd)
	isRecordExist=0
	for row in cursor:
		isRecordExist=1
	#si l'id existe d�ja on fait une modification sinon on cr�e une nouvelle personne
	if(isRecordExist==1):
		cmd="UPDATE personnes SET Name="+str(name)+" WHERE ID="+str(Id)
	else:
		cmd="INSERT INTO personnes(ID,Name) Values("+str(Id)+","+str(name)+")"
	connexion.execute(cmd)
	connexion.commit()
	connexion.close()
	

#l'application nous demande d'entrer l'id et le nom de personne qu'on souhaite ajout�e
Id=raw_input('enter your id')
name=raw_input('enter your name')
insertOrUpdate(Id,name)

sampleNum=0
while True:
        ret, img = video_capture.read()
        if ret: 
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
      			#incrementer sample number
        		sampleNum=sampleNum+1 

			#enregistrer le visage capturer dans le dossier dataset
			cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('Ajout des personnes dans la base de donn�e', img)
        

        if cv2.waitKey(100) & 0xff == ord('q'):
		break
	#Arr�ter apr�s prendre 20 photos
	elif sampleNum>20:
		break

video_capture.release()
cv2.destroyAllWindows()
