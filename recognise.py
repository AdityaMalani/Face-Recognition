import cv2
import numpy as np
import os

subjects = ["", "Elon Musk", "Cristiano Ronaldo"]

def detect(img,scale=1.1):
	 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	 face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
	 if (len(faces) == 0):
	 	return None,None
	 (x, y, w, h) = faces[0]
	 return gray[y:y+w, x:x+h], faces[0]
	 
def train_data(path):
	dirs = os.listdir(path)
	faces=[]
	labels=[]
	for dir_name in dirs:
		if not dir_name.startswith("s"):
			continue
		label = int(dir_name.replace("s", ""))
		subject_dir_path = path + "/" + dir_name
		subject_images_names = os.listdir(subject_dir_path)
		for image_name in subject_images_names:
			image_path = subject_dir_path + "/" + image_name
			image = cv2.imread(image_path)
			cv2.imshow("Training on image...", image)
			cv2.waitKey(100)
			face, rect = detect(image)
			if face is not None:
				faces.append(face)
				labels.append(label)
			cv2.destroyAllWindows()
			cv2.waitKey(1)
			cv2.destroyAllWindows()
	return faces,labels
	
print("Preparing data....")
faces, labels = train_data("train-data")					
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.createLBPHFaceRecognizer()
face_recognizer.train(faces, np.array(labels))

def draw_rect(img,rect):
	(x,y,w,h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
def draw_text(img,text,x,y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	
def predictt(img):
	c = img.copy()
	face, rect = detect(c)
	label= face_recognizer.predict(face)
	label_text = subjects[label[0]]
	draw_rect(c, rect)
	draw_text(c, label_text, rect[0], rect[1]-5)
	return c

print("Predicting Images....")

test1 = cv2.imread('test-data/1.jpeg')
test2 = cv2.imread('test-data/2.jpeg')

predicted_img1 = predictt(test1)
predicted_img2 = predictt(test2)
print("Prediction complete")	
		
cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
