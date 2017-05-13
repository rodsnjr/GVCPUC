import cv2
import numpy as np
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution2D, MaxPooling2D


#input data information
width=300
height=300
num_files_training = 360
num_files_testing  = 156 #total of 516 images per class
num_files = num_files_training + num_files_testing
num_classes = 2

num_epocs = 100

#ret_matrix_training = np.ndarray(shape = (num_files_training*num_classes,width,height)) #shape: (720, 300, 300)
#ret_labels_training = np.ndarray(shape = (num_files_training*num_classes,num_classes))  #shape: (720, 2)
#ret_matrix_testing = np.ndarray(shape = (num_files_testing*num_classes,width,height))   #shape: (312, 300, 300)
#ret_labels_testing = np.ndarray(shape = (num_files_testing*num_classes,num_classes))    #shape: (312, 2)


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
Finally, the data must be reshaped, because the CNN needs a one more parameter, matching the number of channels. 
In this case, we have 1 channel, for the images are in grayscale.
'''
def input_data():
	

	ret_matrix_training = np.ndarray(shape = (num_files_training*num_classes,width,height)) #shape: (720, 300, 300)
	ret_labels_training = np.ndarray(shape = (num_files_training*num_classes,num_classes))  #shape: (720, 2)
	ret_matrix_testing = np.ndarray(shape = (num_files_testing*num_classes,width,height))   #shape: (312, 300, 300)
	ret_labels_testing = np.ndarray(shape = (num_files_testing*num_classes,num_classes))    #shape: (312, 2)
	

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

	ret_matrix_testing = ret_matrix_testing.reshape(ret_matrix_testing.shape[0], 1, width,height)
	ret_matrix_training = ret_matrix_training.reshape(ret_matrix_training.shape[0], 1, width,height)
		
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

	return ret_matrix_training,ret_labels_training,ret_matrix_testing,ret_labels_testing

'''
This function create a model for a CNN, that has for input a 3d data (format (num_channels,width,height))
'''
def create_model(ret_matrix_training):

	model = Sequential()
	
#	model.add(Convolution2D(3,7,7, border_mode= 'same',input_shape=ret_matrix_training.shape[1:]))
#	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(3,3)))
#	model.add(Dropout(0.5))
#
#	model.add(Convolution2D(4,5,5, border_mode= 'same',input_shape=ret_matrix_training.shape[1:]))
#	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(2,2)))
#	model.add(Dropout(0.5))
#
#	model.add(Flatten())
#	model.add(Dense(128))
#	model.add(Activation('relu'))
#	model.add(Dropout(0.5))
#	model.add(Dense(2))
#	model.add(Activation('relu'))
#	model.add(Dropout(0.5))
#	model.add(Dense(num_classes))
#	model.add(Activation('softmax'))

	#grasp detection
#	model.add(Convolution2D(64,2,2, border_mode= 'same',input_shape=ret_matrix_training.shape[1:]))
#	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(2,2)))
#	model.add(Dropout(0.5))
#
#	model.add(Convolution2D(128,2,2, border_mode= 'same',input_shape=ret_matrix_training.shape[1:]))
#	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(2,2)))
#	model.add(Dropout(0.5))
#
#
#	model.add(Convolution2D(128,2,2, border_mode= 'same',input_shape=ret_matrix_training.shape[1:]))
#	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(2,2)))
#	model.add(Dropout(0.5))
#
#
#	model.add(Convolution2D(128,2,2, border_mode= 'same',input_shape=ret_matrix_training.shape[1:]))
#	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(2,2)))
#	model.add(Dropout(0.5))
#
#
#	model.add(Flatten())
#	model.add(Dense(512))
#	model.add(Activation('relu'))
#	model.add(Dropout(0.5))
#	model.add(Dense(2))
#	model.add(Activation('relu'))
#	model.add(Dropout(0.5))
#	model.add(Dense(num_classes))
#	model.add(Activation('softmax'))





#my model, check the ending 'final'
	model.add(Convolution2D(32,3,3, border_mode= 'same',input_shape=ret_matrix_training.shape[1:]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(64,3,3, border_mode= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(128,3,3, border_mode= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))

#	model.add(Convolution2D(64,3,3, border_mode= 'same'))
#	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(2,2)))
#	model.add(Dropout(0.5))

#	model.add(Convolution2D(32,3,3, border_mode= 'same'))
#	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(2,2)))
#	model.add(Dropout(0.5))

	#final
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))


# Marcelo's model 
#	model1 = Sequential()
#	model1.add(Convolution2D(16, 9, 9, border_mode='same', input_shape=ret_matrix_training.shape[1:]))
#	model1.add(Activation('relu'))
#	model1.add(MaxPooling2D(pool_size=(2, 2)))
#	model1.add(Dropout(0.25))               # Desliga x% de vezes o neuronio, evita overfitting
#	model1.add(Convolution2D(32, 7, 7, border_mode='same'))
#	model1.add(Activation('relu'))
#	model1.add(MaxPooling2D(pool_size=(2, 2)))
#	model1.add(Dropout(0.25))
#	model1.add(Convolution2D(16, 7, 7, border_mode='same'))
#	model1.add(Activation('relu'))
#	model1.add(Convolution2D(12, 7, 7, border_mode='same'))
#	model1.add(Activation('relu'))
#	#-----------------------------------
#	model2 = Sequential()
#	model2.add(Convolution2D(20, 7, 7, border_mode='same', input_shape=ret_matrix_training.shape[1:]))
#	model2.add(Activation('relu'))
#	model2.add(MaxPooling2D(pool_size=(2, 2)))
#	model2.add(Dropout(0.25))               # Desliga x% de vezes o neuronio, evita overfitting
#	model2.add(Convolution2D(40, 5, 5, border_mode='same'))
#	model2.add(Activation('relu'))
#	model2.add(MaxPooling2D(pool_size=(2, 2)))
#	model2.add(Dropout(0.25))
#	model2.add(Convolution2D(20, 5, 5, border_mode='same'))
#	model2.add(Activation('relu'))
#	model2.add(Convolution2D(12, 5, 5, border_mode='same'))
#	model2.add(Activation('relu'))
#	#-----------------------------------
#	model3 = Sequential()
#	model3.add(Convolution2D(24, 5, 5, border_mode='same', input_shape=ret_matrix_training.shape[1:]))
#	model3.add(Activation('relu'))
#	model3.add(MaxPooling2D(pool_size=(2, 2)))
#	model3.add(Dropout(0.25))               # Desliga x% de vezes o neuronio, evita overfitting
#	model3.add(Convolution2D(48, 3, 3, border_mode='same'))
#	model3.add(Activation('relu'))
#	model3.add(MaxPooling2D(pool_size=(2, 2)))
#	model3.add(Dropout(0.25))
#	model3.add(Convolution2D(24, 3, 3, border_mode='same'))
#	model3.add(Activation('relu'))
#	model3.add(Convolution2D(12, 3, 3, border_mode='same'))
#	model3.add(Activation('relu'))
#	# MERGE
#	model = Sequential()
#	model.add(Merge([model1, model2, model3], mode='ave', concat_axis=1))


	#final
#	model.add(Flatten())
#	model.add(Dense(128))
#	model.add(Activation('relu'))
#	model.add(Dropout(0.5))
#	model.add(Dense(num_classes))
#	model.add(Activation('softmax'))

	return model
	#print model.summary()
	'''
	Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 32, 10, 10)    320         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 32, 10, 10)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 5, 5)      0           activation_1[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 32, 5, 5)      0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 64, 5, 5)      18496       dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 64, 5, 5)      0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 64, 2, 2)      0           activation_2[0][0]               
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 64, 2, 2)      0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 128, 2, 2)     73856       dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 128, 2, 2)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 128, 1, 1)     0           activation_3[0][0]               
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 128, 1, 1)     0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 128)           0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           16512       flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 128)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 128)           0           activation_4[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 2)             258         dropout_4[0][0]                  
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 2)             0           dense_2[0][0]                    
====================================================================================================
Total params: 109442
____________________________________________________________________________________________________
	'''
'''
Function to compile and train the CNN. Note that the batch_size must be multiple by num_files_training. 
'''
def compile_model(model,ret_matrix_training,ret_labels_training,ret_matrix_testing,ret_labels_testing):
	model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	#model.fit([ret_matrix_training,ret_matrix_training,ret_matrix_training],ret_labels_training, batch_size=36, nb_epoch=num_epocs,show_accuracy=True, validation_data=([ret_matrix_testing,ret_matrix_testing,ret_matrix_testing],ret_labels_testing))
	model.fit(ret_matrix_training,ret_labels_training, batch_size=36, nb_epoch=num_epocs,show_accuracy=True, validation_data=(ret_matrix_testing,ret_labels_testing))


################################main###################################
start = time.time()

#normalize_figs(width,height)
ret_matrix_training,ret_labels_training,ret_matrix_testing,ret_labels_testing = input_data()
model = create_model(ret_matrix_training)
compile_model(model,ret_matrix_training,ret_labels_training,ret_matrix_testing,ret_labels_testing)

end = time.time()

print "Time elapsed:",(end - start)
