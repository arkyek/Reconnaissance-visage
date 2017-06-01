# -*- coding: utf-8 -*-

import cv2, sys, numpy, os
size = 1
fichierXML = 'haarcascade_frontalface_default.xml'
data_Faces = 'DATA'


try:
    nom_personne = sys.argv[1]
except:
    print("Vous devez entrer un nom")
    sys.exit(0)

path = os.path.join(data_Faces, nom_personne)
if not os.path.isdir(path):
    os.mkdir(path)
(im_width, im_height) = (112, 92)

haar_cascade = cv2.CascadeClassifier(fichierXML)
camera = cv2.VideoCapture(0)

# Généré le nom pour les images
pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
     if n[0]!='.' ]+[0])[-1] + 1

# Message du début
print("\n\033[94mLe programme va enregistrer 20 images d'apprentissage. \
Veuillez fixer votre tête proche de la caméra et changer votre expression aprés chaque image..\033[0m\n")

#l'application poursuit jusqu'à il prend 20 images d'un visage
count = 0
pause = 0
count_max = 20
while count < count_max:

    #boucler jusqu'a la caméra foncionne
    rval = False
    while(not rval):
        #mettre l'image du caméra dans la fenêtre
        (rval, frame) = camera.read()
        if(not rval):
            print("Impossible d'ouvrir la caméra. Essayer à nouveau...")

    # obtenir la taille de l'image
    height, width, channels = frame.shape

    # Flip la fêntre
    frame = cv2.flip(frame, 1, 0)

    # Convertir l'image en niveau du gris (grayscale,echelle de gris)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # réduire la vitesse
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    # Détécter les visages
    visages = haar_cascade.detectMultiScale(mini)

    # on prend en compte juste les grands visages (les grand carrés), parfois il detect des petits carrés qui sont pas du visage
    visages = sorted(visages, key=lambda x: x[3])
    if visages:
        visage_i = visages[0]
        (x, y, w, h) = [v * size for v in visage_i]

        visage = gray[y:y + h, x:x + w]
        visage_resize = cv2.resize(visage, (im_width, im_height))

        # Dessinez un rectangle autour du visage détecté et le nom entré
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, nom_personne, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
            1,(0, 255, 0))

        # Si le rectangle est petit
        if(w * 6 < width or h * 6 < height):
            print("Visage très petit")
        else:

            # enregistrer que les visages dans les grands rectangle afin de garantir avoir des bons images dans la base de données
            if(pause == 0):

                print("Enregistrement des images d'apprentissage "+str(count+1)+"/"+str(count_max))

                # enregistrer le visage pris
                cv2.imwrite("%s/%s.png" % (path, pin), visage_resize)

                pin += 1
                count += 1

                pause = 1

    if(pause > 0):
        pause = (pause + 1) % 5

    ##ouvrir la fenêtre avec le titre "Enregistrement des visages"
    cv2.imshow('Enregistrement des visages', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
