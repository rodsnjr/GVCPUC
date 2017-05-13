import cv2
import numpy as np
import os
import time


#input data information
width=300
height=300
num_files_training = 360
num_files_testing  = 156 #total of 516 images per class
num_files = num_files_training + num_files_testing
num_classes = 2


ret_matrix_training = np.ndarray(shape = (num_files_training*num_classes,width,height)) #shape: (720, 300, 300)
ret_labels_training = np.ndarray(shape = (num_files_training*num_classes,num_classes))  #shape: (720, 2)
ret_matrix_testing = np.ndarray(shape = (num_files_testing*num_classes,width,height))   #shape: (312, 300, 300)
ret_labels_testing = np.ndarray(shape = (num_files_testing*num_classes,num_classes))    #shape: (312, 2)




#file paths
path = "tf_files/" #root directory from the classes
indoor_files = os.listdir(path + "Indoor_environment/")
door_files = os.listdir(path + "Doors/")

'''
"Normalize" the input data. This function will change the resolution of each picture from both clases
and transform them from RGB (3 channels) to grayscale (1 channel).
ALL the pictures will be overwritten!!
'''
def normalize_figs(W=300,H=300):
	for n in range(num_files):
		img = cv2.imread(path + "Indoor_environment/"+ indoor_files[n])
		print path + "Indoor_environment/"+ indoor_files[n]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img,(W,H))
		cv2.imwrite(path + "Indoor_environment/"+ indoor_files[n], img)

	for n in range(num_files):		
		img = cv2.imread(path + "Doors/"+ door_files[n])
		print path + "Doors/"+ door_files[n]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img,(W,H))
		cv2.imwrite(path + "Doors/"+ door_files[n], img)


'''
This function will read all the normalized data from the specified directories. It's necessary convert again 
to grayscale because when the image is loaded, the default number of chennels is 3. 
Each picture will be loaded to a 3d array, being the first dimension the number of files that will be stored.
We will train the network with aprox 70% of the data and test it with 30%. These data structures are divided in 2 
3d structures each.
'''
def input_data():
	print "LOADNIG DATA, PLEASE WAIT"
	#class 0: pictures of indoor environments without doors appearing
	counter = 0  

	for n in range(0,num_files_training):
		image = cv2.imread(path + "Indoor_environment/"+ indoor_files[n])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for j in range(0,height):
			for i in range(0,width ):
				ret_matrix_training[n][j][i] = image[j][i]
		counter = counter + 1

	for n in range(0,num_files_testing):
		image = cv2.imread(path + "Indoor_environment/"+ indoor_files[counter])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for j in range(0,height):
			for i in range(0,width):
				ret_matrix_testing[n][j][i] = image[j][i]
		counter = counter + 1		



	#class 1: pictures of doors and indoor environments with doors appearing
	counter = 0  
	for n in range(num_files_training,num_files_training*2):
		image = cv2.imread(path + "Doors/"+ door_files[counter])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for j in range(0,height):
			for i in range(0,width ):
				ret_matrix_training[n][j][i] = image[j][i]
		counter = counter + 1

	for n in range(num_files_testing,num_files_testing*2):
		image = cv2.imread(path + "Doors/"+ door_files[counter])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for j in range(0,height):
			for i in range(0,width ):
				#print j, " ",i
				ret_matrix_testing[n][j][i] = image[j][i]
		counter = counter + 1




#	print "\n\ntraining:"
#	for n in range(0,num_files_training*num_classes):
#		for j in range(0,height):
#			for i in range(0,width ):
#				print ret_matrix_training[n][j][i]
#		print "\n"
#
#	print "\n\ntesting:"
#	for n in range(0,num_files_testing*num_classes):
#		for j in range(0,height):
#			for i in range(0,width ):
#				print ret_matrix_testing[n][j][i]
#		print "\n" 


#	print "training:"
#	for n in range(0,num_files_training*num_classes):
#		for j in range(0,height):
#			for i in range(0,width ):
#				print ret_matrix_training[n][j][i]
#		print "\n"
#
#	print "testing:"
#	for n in range(0,num_files_testing*num_classes):
#		for j in range(0,height):
#			for i in range(0,width ):
#				print ret_matrix_testing[n][j][i]
#		print "\n"



#for only one dataset, without separating in training and testing.
#	p = 0
#	for n in range(0,num_files):
#		image = cv2.imread(path + "Indoor_environment/"+ indoor_files[n])
#		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#		if(n)
#
#		for j in range(0,height):
#			for i in range(0,width ):
#				ret_matrix[n][j][i] = image[j][i]
#		p = p + 1
#
#	#class 1: pictures of doors and indoor environments with doors appearing
#	for n in range(0,num_files):
#		image = cv2.imread(path + "Doors/"+ door_files[n])
#		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#		for j in range(0,height):
#			for i in range(0,width ):
#				ret_matrix[p][j][i] = image[j][i]
#		p = p + 1
#
#

#ret_matrix[n] = ret_matrix[n].astype('float32')
#ret_matrix[n] /= 255


	for n in range(0,num_files_training*2):
		if n < num_files_training:
			ret_labels_training[n][0]=1
			ret_labels_training[n][1]=0
		else:
			ret_labels_training[n][0]=0
			ret_labels_training[n][1]=1

	for n in range(0,num_files_testing*2):
		if n < num_files_testing:
			ret_labels_testing[n][0]=1
			ret_labels_testing[n][1]=0
		else:
			ret_labels_testing[n][0]=0
			ret_labels_testing[n][1]=1
	print "DATA LOADED"

	#for n in range(0,num_files_testing*2):
	#	print n, ret_labels_testing[n]



start = time.time()

#normalize_figs(width,height)
input_data()

end = time.time()

print "Time elapsed:",(end - start)
