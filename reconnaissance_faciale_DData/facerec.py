# -*- coding: utf-8 -*-


import cv2
import sys
import numpy
import os

dataFaces = 'DATA'


######### Partie 1 créer un LBPHF Recognizer############################
########################################################################

print('Detection des visage en cours...')

#création de la liste des images et des noms de personnes
images = []
lables = []
noms = {}
id = 0

#Obtenez les dossiers contenant les données
for (subdirs, dossiers, files) in os.walk(dataFaces):
	#Bouclez chaque dossier 
	for dossier in dossiers:
		noms[id] = dossier
		cheminPersonne = os.path.join(dataFaces, dossier)
		
		#Boucler chaque image dans le dossier
		for nomFichier in os.listdir(cheminPersonne):
			#négliger les non-images si il existe
			n_fichier , extension_fichier = os.path.splitext(nomFichier)
			if(extension_fichier.lower() not in ['.png','.jpg','.jpeg','.gif','.pgm']):
				print("Négliger le "+nomFichier+", format incorrecte")
				continue
			chemin = cheminPersonne + "/" + nomFichier
			lable = id
	
			#ajouter à training data (dans les listes)
			images.append(cv2.imread(chemin, 0))
			lables.append(int(lable))
		id +=1
(im_width, im_height) = (112, 92)

#création d'un tableau Numpy à partir des 2 liste images et lables
(images, lables) = [numpy.array(lis) for lis in [images, lables]]
	
#trainer le model avec OpenCV à partir des images
model = cv2.createLBPHFaceRecognizer()  #cv2.face.createFisherFaceRecognizer()	#createEigenFaceRecognizer()
model.train(images, lables)


##### Partie 2 utiliser le LBPHF Recognizer#############################
########################################################################

#charger le fichier xml 
fichierXML = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(fichierXML)

#utiliser la caméra 0
camera = cv2.VideoCapture(0)

while True:
	#tourner la boucle jusqu'a que la caméra marche
	img = False
	while (not img):
		#metre l'image du caméra dans la fenêtre 'fenetre'
		img , fenetre = camera.read()
		if(not img):
			print("Impossible d'ouvrir la caméra. Essayer à nouveau...")
	
	#convertir l'image en niveau du gris (grayscale,echelle de gris)
	gray = cv2.cvtColor(fenetre, cv2.COLOR_BGR2GRAY)

	#augementer la valeur du size pour accélérer la detection
	size = 1
	# Redimensionnez l'image pour accélérer la détection
	mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

	#Dessinez des rectangles autour de chaque visage détecté
	faces = faceCascade.detectMultiScale(mini)
	for i in range(len(faces)):
		face_i = faces[i]
		
		## Coordonnées du visage après échelonnement par 'size'
		(x, y, w, h) = [v * size for v in face_i] #Échelle de la sauvegarde de formes
		face = gray[y:y + h, x:x + w]
		face_resize = cv2.resize(face, (im_width, im_height))

		#identifier le visage
		prediction = model.predict(face_resize)
		cv2.rectangle(fenetre, (x, y), (x + w, y + h),(0,255,0),3)
	
		#return le nom du visage détécté
		if prediction[1]<500:
			cv2.putText(fenetre,'%s' %(noms[prediction[0]]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
		else:
			cv2.putText(fenetre, 'Inconnue', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

	#ouvrir la fenêtre avec le titre "Identification des visages"
	cv2.imshow("Identification des visages" , fenetre)

	#ECHAP pour quitter
        k = cv2.waitKey(30) & 0xff
        if k == 27: #the Esc key
		break

		
