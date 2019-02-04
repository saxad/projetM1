import cv2
import matplotlib.pyplot as plt
import argparse


argObj = argparse.ArgumentParser()
argObj.add_argument("-i", required = True, help="entrer le chemin verre la copie à corriger")

arg = vars(argObj.parse_args())

print(arg)

image = cv2.imread(arg["i"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray,(5,5), 0)

edged = cv2.Canny(blurred, 75, 200)
plt.imshow(edged, cmap='gray')
plt.show()
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

if len(cnts) > 0:
	
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	for c in cnts:
		
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			docCnt = approx
			break
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
