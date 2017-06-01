# -*- coding: cp1252 -*-

import cv2
import sqlite3

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

# fonction qui ajoute ou modifie les personnes dans la base de données
def insertOrUpdate(Id,name):
	#connexion a la base de donnée "faceDB"
	connexion=sqlite3.connect("facesDB")
	#requête qui cherche l'id entré pour verifier s'il existe déja dans la BD
	cmd="SELECT * FROM personnes WHERE ID="+str(Id)
	cursor=connexion.execute(cmd)
	isRecordExist=0
	for row in cursor:
		isRecordExist=1
	#si l'id existe déja on fait une modification sinon on crée une nouvelle personne
	if(isRecordExist==1):
		cmd="UPDATE personnes SET Name="+str(name)+" WHERE ID="+str(Id)
	else:
		cmd="INSERT INTO personnes(ID,Name) Values("+str(Id)+","+str(name)+")"
	connexion.execute(cmd)
	connexion.commit()
	connexion.close()
	

#l'application nous demande d'entrer l'id et le nom de personne qu'on souhaite ajoutée
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

                cv2.imshow('Ajout des personnes dans la base de donnée', img)
        

        if cv2.waitKey(100) & 0xff == ord('q'):
		break
	#Arréter après prendre 20 photos
	elif sampleNum>20:
		break

video_capture.release()
cv2.destroyAllWindows()
