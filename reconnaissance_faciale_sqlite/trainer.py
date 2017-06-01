# -*- coding: cp1252 -*-

import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()  #cv2.face.createFisherFaceRecognizer()	#createEigenFaceRecognizer()
path="dataSet"

def getImagesWithID(path):
	#Obtenir le chemin de tous les fichiers dans le dossier
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

	#crée une liste vide des visages et des IDs
	faces=[]
	IDs=[]
	#now looping through all the image paths and loading the Ids and the images
	#on boucle maintenant tous les chemins d'images et on charge les Ids et les images
	for imagePath in imagePaths:
		#Chargement de l'image et la convertir en échelle de gris
		faceImage=Image.open(imagePath).convert('L')
		 #convertir le  PIL image au tableau numpy
		faceNp=np.array(faceImage,'uint8')
		#obtenir l'Id de image
		Id=int(os.path.split(imagePath)[-1].split(".")[1])
		faces.append(faceNp)
		IDs.append(Id)
		cv2.imshow("training",faceNp)
		cv2.waitKey(10)
	return IDs,faces

IDs,faces=getImagesWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('trainner/trainner.yml')
cv2.destroyAllWindows()

