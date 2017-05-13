import cv2
import numpy as np
import os

width=300
height=300
num_files = 516*2


path = "tf_files/" #root directory from the classes
indoor_files = os.listdir(path + "Indoor_environment/")
door_files = os.listdir(path + "detection2/Doors/") 
labels_portas = path + "detection2/labels_portas/"
label_indoor = path + "detection2/labels_indoor/"

def portas():

	#com=0
	#sem =0
	for n in range(num_files):
	
		string = path + "detection2/Doors/"+ door_files[n]
		#print string
		new_string = string.split('.')
		if(len(new_string) == 2):
	#		sem = sem+1
	#	elif(len(new_string) == 3):
	#		com = com + 1
	#	else:
	#		print string
	#print "sem:", sem
	#print "com", com
			img = cv2.imread(string)
		
		
			#Gets the shape of the original image and creates a new one 
			height, width, channels = img.shape
			blank_image = np.zeros((height,width,1), np.uint8)
		
			#treats the txt file with the coordinates
			txt_file = open(string + ".txt","r")
		
			#read the text file and draws on the new image
			text=  txt_file.readline()
			while(len(text) > 0):
				coordenates= text.split(',')
				cv2.rectangle(blank_image,(int(coordenates[0]),int(coordenates[1])),(int(coordenates[2]),int(coordenates[3])),(255,255,255),-1)
				text=  txt_file.readline()
			print labels_portas + door_files[n]+"_GT.jpg"
			#saves the image..
			cv2.imwrite(labels_portas + door_files[n]+"_GT.jpg",blank_image)
	
def indoor():
	for n in range(516):
		string = path + "detection2/Indoor_environment/"+ indoor_files[n]
		print string
		img = cv2.imread(string)
		height, width, channels = img.shape
		blank_image = np.zeros((height,width,1), np.uint8)
		cv2.imwrite(label_indoor + indoor_files[n]+"_GT.jpg",blank_image)




indoor()