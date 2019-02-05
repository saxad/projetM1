import cv2
import matplotlib.pyplot as plt
import argparse
import imutils
from imutils.perspective  import four_point_transform


argObj = argparse.ArgumentParser()
argObj.add_argument("-i", required = True, help="entrer le chemin verre la copie à corriger")

arg = vars(argObj.parse_args())

print(arg)

image = cv2.imread(arg["i"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray,(5,5), 0)

edged = cv2.Canny(blurred, 75, 200)






#recupérer les contours externes de la page puis garde ses 4 pts
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)



cnts = imutils.grab_contours(cnts)
docCnt = None

if len(cnts) > 0:

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        #si on trouve les 4 points c'est qu'on trouvé la photo
        if len(approx) == 4:
            docCnt = approx
            break

paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

thresh = cv2.threshold(warped, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []


plt.imshow(paper, cmap='gray')
plt.show()
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)
#trier les contours de la question de haut en bas, puis initialiser le nombre total de réponses correctes
questionCnts = contours.sort_contours(questionCnts,
	method="top-to-bottom")[0]
correct = 0
