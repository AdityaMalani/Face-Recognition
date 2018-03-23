import cv2
import time


test1 = cv2.imread('images/4.jpg')

haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def detect_faces(f_cascade,colourimg,scale=1.1):
	copy = colourimg.copy()
	gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
	#print("reached1")
	faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=scale,minNeighbors=5);
	#print("reached2")
	for (x,y,w,h) in faces :
		#print("reached3")
		cv2.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),2)
	return copy			
	
faces_detected_img = detect_faces(haar_face_cascade, test1)

cv2.imshow('Test Imag', faces_detected_img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
