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
