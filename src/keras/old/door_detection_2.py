import cv2
import numpy as np
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, RMSprop, Adam, Adamax, Nadam
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint








MyOptimizer = Nadam() #Adadelta()
def MAE(y_true, y_pred):
	return K.mean(K.abs(y_pred - y_true))    

def MSE(y_true, y_pred):
	# Este eh o MSE do paper 2016
	#return K.sqrt( K.mean(K.square(y_pred - y_true) ) )
	# Este eh o MSE verdadeiro
	return K.mean(K.square(y_pred - y_true) ) 


def euclidean_distance(y_true, y_pred):
	return K.sqrt(K.sum(K.square(y_true - y_pred), axis=1, keepdims=True))

#input data information
width=300
height=300
num_files_training = 360
num_files_testing  = 156 #total of 516 images per class
num_files = num_files_training + num_files_testing
num_classes = 2

num_epocs = 50

#ret_matrix_training = np.ndarray(shape = (num_files_training*num_classes,width,height)) #shape: (720, 300, 300)
#ret_labels_training = np.ndarray(shape = (num_files_training*num_classes,num_classes))  #shape: (720, 2)
#ret_matrix_testing = np.ndarray(shape = (num_files_testing*num_classes,width,height))   #shape: (312, 300, 300)
#ret_labels_testing = np.ndarray(shape = (num_files_testing*num_classes,num_classes))    #shape: (312, 2)


#file paths
path = "tf_files/" #root directory from the classes
indoor_files = os.listdir(path + "detection2/Indoor_environment/")
door_files = os.listdir(path + "detection2/Doors/")


labels_portas_path = path + "detection2/labels_portas/"
label_indoor_path = path + "detection2/labels_indoor/"

labels_portas_files = os.listdir(labels_portas_path)
labels_indoor_files = os.listdir(label_indoor_path)

#'''
#"Normalize" the input data. This function will change the resolution of each picture from both clases
#and transform them from RGB (3 channels) to grayscale (1 channel).
#ALL the pictures will be overwritten!!
#'''
#def normalize_figs(W=300,H=300):
#	for n in range(num_files):
#		img = cv2.imread(path + "Indoor_environment/"+ indoor_files[n])
#		print path + "Indoor_environment/"+ indoor_files[n]
#		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#		img = cv2.resize(img,(W,H))
#		cv2.imwrite(path + "Indoor_environment/"+ indoor_files[n], img)
#
#	for n in range(num_files):		
#		img = cv2.imread(path + "Doors/"+ door_files[n])
#		print path + "Doors/"+ door_files[n]
#		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#		img = cv2.resize(img,(W,H))
#		cv2.imwrite(path + "Doors/"+ door_files[n], img)
#
#
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
	ret_labels_training = np.ndarray(shape = (num_files_training*num_classes,75,75)) #shape: (720, 300, 300)
	ret_matrix_testing = np.ndarray(shape = (num_files_testing*num_classes,width,height))   #shape: (312, 300, 300)
	ret_labels_testing = np.ndarray(shape = (num_files_testing*num_classes,75,75))   #shape: (312, 300, 300)
	
#
	print "LOADNIG DATA, PLEASE WAIT"

	#class 0: pictures of indoor environments without doors appearing
	counter = 0  

	for n in range(0,num_files_training):
		image = cv2.imread(path + "detection2/Indoor_environment/"+ indoor_files[n])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for j in range(0,height):
			for i in range(0,width ):
				ret_matrix_training[n][j][i] = image[j][i]
		counter = counter + 1

	for n in range(0,num_files_testing):
		image = cv2.imread(path + "detection2/Indoor_environment/"+ indoor_files[counter])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for j in range(0,height):
			for i in range(0,width):
				ret_matrix_testing[n][j][i] = image[j][i]
		counter = counter + 1		



	#class 1: pictures of doors and indoor environments with doors appearing
	counter = 0  
	for n in range(num_files_training,num_files_training*2):
		image = cv2.imread(path + "detection2/Doors/"+ door_files[counter])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for j in range(0,height):
			for i in range(0,width ):
				ret_matrix_training[n][j][i] = image[j][i]
		counter = counter + 1

	for n in range(num_files_testing,num_files_testing*2):
		image = cv2.imread(path + "detection2/Doors/"+ door_files[counter])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for j in range(0,height):
			for i in range(0,width ):
				#print j, " ",i
				ret_matrix_testing[n][j][i] = image[j][i]
		counter = counter + 1



	ret_matrix_testing = ret_matrix_testing.reshape(ret_matrix_testing.shape[0], 1, width,height)
	ret_matrix_training = ret_matrix_training.reshape(ret_matrix_training.shape[0], 1, width,height)



##############labels###########################
	counter = 0
	for n in range(0,num_files_training):
		image = cv2.imread(path + "detection2/labels_indoor/"+ labels_indoor_files[n])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image,(75,75))
		for j in range(0,75):
			for i in range(0,75 ):
				ret_labels_training[n][j][i] = image[j][i]
		counter = counter + 1

	for n in range(0,num_files_testing):
		image = cv2.imread(path + "detection2/labels_indoor/"+ labels_indoor_files[counter])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image,(75,75))
		for j in range(0,75):
			for i in range(0,75):
				ret_labels_testing[n][j][i] = image[j][i]
		counter = counter + 1		




	counter = 0
	for n in range(num_files_training,num_files_training*2):
		image = cv2.imread(path + "detection2/labels_portas/"+ labels_portas_files[counter])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image,(75,75))
		for j in range(0,75):
			for i in range(0,75 ):
				ret_labels_training[n][j][i] = image[j][i]
		counter = counter + 1

	for n in range(num_files_testing,num_files_testing*2):
		image = cv2.imread(path + "detection2/labels_portas/"+ labels_portas_files[counter])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image,(75,75))
		for j in range(0,75):
			for i in range(0,75):
				ret_labels_testing[n][j][i] = image[j][i]
		counter = counter + 1

	ret_labels_testing = ret_labels_testing.reshape(ret_labels_testing.shape[0], 1, 75,75)
	ret_labels_training = ret_labels_training.reshape(ret_labels_training.shape[0], 1, 75,75)

	print "DATA LOADED"
#
	return ret_matrix_training,ret_labels_training,ret_matrix_testing,ret_labels_testing

'''
This function create a model for a CNN, that has for input a 3d data (format (num_channels,width,height))
'''
def create_model(ret_matrix_training):

	model = Sequential()


# Marcelo's model 
	model1 = Sequential()
	model1.add(Convolution2D(16, 9, 9, border_mode='same', input_shape=ret_matrix_training.shape[1:]))
	model1.add(Activation('relu'))
	model1.add(MaxPooling2D(pool_size=(2, 2)))
	model1.add(Dropout(0.25))               # Desliga x% de vezes o neuronio, evita overfitting
	model1.add(Convolution2D(32, 7, 7, border_mode='same'))
	model1.add(Activation('relu'))
	model1.add(MaxPooling2D(pool_size=(2, 2)))
	model1.add(Dropout(0.25))
	model1.add(Convolution2D(16, 7, 7, border_mode='same'))
	model1.add(Activation('relu'))
	model1.add(Convolution2D(12, 7, 7, border_mode='same'))
	model1.add(Activation('relu'))
	#-----------------------------------
	model2 = Sequential()
	model2.add(Convolution2D(20, 7, 7, border_mode='same', input_shape=ret_matrix_training.shape[1:]))
	model2.add(Activation('relu'))
	model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Dropout(0.25))               # Desliga x% de vezes o neuronio, evita overfitting
	model2.add(Convolution2D(40, 5, 5, border_mode='same'))
	model2.add(Activation('relu'))
	model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Dropout(0.25))
	model2.add(Convolution2D(20, 5, 5, border_mode='same'))
	model2.add(Activation('relu'))
	model2.add(Convolution2D(12, 5, 5, border_mode='same'))
	model2.add(Activation('relu'))
	#-----------------------------------
	model3 = Sequential()
	model3.add(Convolution2D(24, 5, 5, border_mode='same', input_shape=ret_matrix_training.shape[1:]))
	model3.add(Activation('relu'))
	model3.add(MaxPooling2D(pool_size=(2, 2)))
	model3.add(Dropout(0.25))               # Desliga x% de vezes o neuronio, evita overfitting
	model3.add(Convolution2D(48, 3, 3, border_mode='same'))
	model3.add(Activation('relu'))
	model3.add(MaxPooling2D(pool_size=(2, 2)))
	model3.add(Dropout(0.25))
	model3.add(Convolution2D(24, 3, 3, border_mode='same'))
	model3.add(Activation('relu'))
	model3.add(Convolution2D(12, 3, 3, border_mode='same'))
	model3.add(Activation('relu'))
	# MERGE
	model = Sequential()
	model.add(Merge([model1, model2, model3], mode='ave', concat_axis=1))

	model.add(Convolution2D(1, 1, 1, border_mode='same'))
	model.add(Activation('relu'))
	print model.summary()

	return model
	'''
	Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 16, 300, 300)  1312        convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 16, 300, 300)  0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 16, 150, 150)  0           activation_1[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 16, 150, 150)  0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 150, 150)  25120       dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 32, 150, 150)  0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 32, 75, 75)    0           activation_2[0][0]               
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 32, 75, 75)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 16, 75, 75)    25104       dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 16, 75, 75)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 12, 75, 75)    9420        activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 12, 75, 75)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 20, 300, 300)  1000        convolution2d_input_2[0][0]      
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 20, 300, 300)  0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 20, 150, 150)  0           activation_5[0][0]               
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 20, 150, 150)  0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 40, 150, 150)  20040       dropout_3[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 40, 150, 150)  0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 40, 75, 75)    0           activation_6[0][0]               
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 40, 75, 75)    0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 20, 75, 75)    20020       dropout_4[0][0]                  
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 20, 75, 75)    0           convolution2d_7[0][0]            
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 12, 75, 75)    6012        activation_7[0][0]               
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 12, 75, 75)    0           convolution2d_8[0][0]            
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 24, 300, 300)  624         convolution2d_input_3[0][0]      
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 24, 300, 300)  0           convolution2d_9[0][0]            
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 24, 150, 150)  0           activation_9[0][0]               
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 24, 150, 150)  0           maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
convolution2d_10 (Convolution2D) (None, 48, 150, 150)  10416       dropout_5[0][0]                  
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 48, 150, 150)  0           convolution2d_10[0][0]           
____________________________________________________________________________________________________
maxpooling2d_6 (MaxPooling2D)    (None, 48, 75, 75)    0           activation_10[0][0]              
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 48, 75, 75)    0           maxpooling2d_6[0][0]             
____________________________________________________________________________________________________
convolution2d_11 (Convolution2D) (None, 24, 75, 75)    10392       dropout_6[0][0]                  
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 24, 75, 75)    0           convolution2d_11[0][0]           
____________________________________________________________________________________________________
convolution2d_12 (Convolution2D) (None, 12, 75, 75)    2604        activation_11[0][0]              
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 12, 75, 75)    0           convolution2d_12[0][0]           
____________________________________________________________________________________________________
convolution2d_13 (Convolution2D) (None, 1, 75, 75)     13          merge_1[0][0]                    
____________________________________________________________________________________________________
activation_13 (Activation)       (None, 1, 75, 75)     0           convolution2d_13[0][0]           
====================================================================================================

	'''

'''
Function to compile and train the CNN. Note that the batch_size must be multiple by num_files_training. 
'''
def compile_model(model,ret_matrix_training,ret_labels_training,ret_matrix_testing,ret_labels_testing):
	#model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	model.compile(loss=euclidean_distance,optimizer=MyOptimizer,metrics=[MAE,MSE])


	filepath="model-{epoch:03d}-{val_MAE:.4f}-{val_MSE:.4f}.h5w"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', save_weights_only=True)
	# para parar quando nao hover mudanca no val_loss
	earlyStopping= EarlyStopping(monitor='val_loss', patience=35, verbose=0, mode='auto')
	callbacks_list = [checkpoint, earlyStopping]
	model.fit([ret_matrix_training,ret_matrix_training,ret_matrix_training],ret_labels_training, batch_size=36, nb_epoch=num_epocs,shuffle=True,show_accuracy=True,verbose=1, validation_data=([ret_matrix_testing,ret_matrix_testing,ret_matrix_testing],ret_labels_testing),callbacks=callbacks_list)
	model.save_weights("model.h5w")


#################################main###################################
start = time.time()

#normalize_figs(width,height)
ret_matrix_training,ret_labels_training,ret_matrix_testing,ret_labels_testing = input_data()
model = create_model(ret_matrix_training)
compile_model(model,ret_matrix_training,ret_labels_training,ret_matrix_testing,ret_labels_testing)

end = time.time()

print "Time elapsed:",(end - start)
