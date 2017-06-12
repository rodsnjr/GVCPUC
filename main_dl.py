#!-*- conding: utf8 -*-
"""
    Testing a Simple SVM with HoG Features
"""
from gvc.general.files.loaders import ImageLoader
from gvc.classifiers.model import SVC
from gvc.classifiers.bow import BOW

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
print sys.argv

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Input
import argparse
import cv2

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.np_utils import to_categorical


def load_imgs(PATH,file_name,res,chennels):
    #print file_name
    file = open(PATH+file_name,"r")
    lines = file.readlines()
    file.close()

    images = np.ndarray(shape = (len(lines)*3,res,res,chennels))
    labels = np.ndarray(shape = (len(lines)*3,1))

    counter_img = 0
    counter_lbl = 0
    #input de dados para o treinamento. 
    for x in range(0,len(lines)): 
        buffer_str = lines[x].split(',')
        img = cv2.imread(PATH + buffer_str[0])

        if(chennels==1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width, channels = img.shape
        
        buffer = buffer_str[3].splitlines()
        buffer_str[3] = buffer[0]
        #analisa os labels..
        for i in range(1,4):
            if(buffer_str[i]=="door"):
                labels[counter_lbl]=1
            if(buffer_str[i]=="indoors"):
                labels[counter_lbl]=0
            if(buffer_str[i]=="stairs"):
                labels[counter_lbl]=0
            counter_lbl+=1
    
        #separa as imagens e coloca uma em casa posicao..
        buffer_img = img[0:height,0:width/3]
        buffer_img = cv2.resize(buffer_img, (res, res)) 
        images[counter_img] = buffer_img
        counter_img+=1
    
        buffer_img = img[0:height,width/3:2*width/3]
        buffer_img = cv2.resize(buffer_img, (res, res)) 
        images[counter_img] =buffer_img
        counter_img+=1
    
        buffer_img = img[0:height,2*width/3:width]
        buffer_img = cv2.resize(buffer_img, (res, res)) 
        images[counter_img] =buffer_img
        counter_img+=1
    
    return images,labels


def parse():
    " Parse the args to launch this app "
    parser = argparse.ArgumentParser(description='Read images from and save the labels on csv.')
    
    parser.add_argument('--dir', type=str, help='Path of the images')
    parser.add_argument('--training', type=str, help="Name of the training csv")
    parser.add_argument('--testing', type=str, help="Name of the testing csv")
    
    args = parser.parse_args()
    return args

def local_loader(path_name,filename):
    loader = ImageLoader(path_name)

    flow = loader.crop_flow_from_csv(filename)
    # test_flow = train_flow.extract_flow(1000)
    return flow

#def VGG_16(weights_path=None):
#    model = Sequential()
#    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
#    model.add(Convolution2D(64, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(64, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(128, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(128, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(256, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(256, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(256, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(Flatten())
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(1000, activation='softmax'))
#
#    if weights_path:
#        model.load_weights(weights_path)
#
#    return model

if __name__ == "__main__":

    args = parse()

    #flow_training= local_loader(args.dir,args.training)
    #flow_testing = local_loader(args.dir,args.testing)
    
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    print "---------------------------------------------"
    print "---------------------------------------------"
    print "Formato das imagens channel_last: (300,300,3)"
    print "---------------------------------------------"
    print "---------------------------------------------"

    training_images,training_labels = load_imgs(args.dir,args.training,300,3)
    testing_images,testing_labels = load_imgs(args.dir,args.testing,300,3)


    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')
    print  model_vgg16_conv.summary()


    input = Input(shape=training_images[0].shape ,name = 'image_input')
    
    output_vgg16_conv = model_vgg16_conv(input)
    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(2, activation='softmax', name='predictions')(x)

    #Create your own model 
    my_model = Model(input=input, output=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()

    training_labels_OH = to_categorical(training_labels)
    testing_labels_OH = to_categorical(testing_labels)

    my_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    # Depois que carrega a rede ...
    my_model.fit(training_images,training_labels_OH, batch_size=16, epochs=50,validation_data=(testing_images,testing_labels_OH))



    #Use the generated model 
    #output_vgg16_conv = model_vgg16_conv(input)
    #top_model = Sequential()
    #top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #top_model.add(Dense(256, activation='relu'))
    #top_model.add(Dropout(0.5))
    #top_model.add(Dense(1, activation='sigmoid'))
    
    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    #top_model.load_weights(top_model_weights_path)





#    #print training_images[0].shape
#    #input_tensor = Input(shape=training_images[0].shape) #(numero de imagens, resolucao da imagem)...
#    input_tensor = Input(tensor=training_images, shape=training_images.shape)
#    base_model = InceptionV3(input_tensor=input_tensor,weights='imagenet', include_top=False) # importa o inception com o input acima.
#    # add a global spatial average pooling layer
#    x = base_model.output
#    x = GlobalAveragePooling2D()(x) #eh tipo um fully connected, link: https://www.quora.com/What-is-global-average-pooling
#    x = Dense(1024, activation='relu')(x)
#    predictions = Dense(2, activation='softmax')(x) #ultima camada da rede, com duas classes. 
#    model = Model(input=base_model.input, output=predictions)
#    #seta as camadas 172 para cima para serem treinadas. AS outras nao serao modificadas..
#    for layer in model.layers[:172]:
#       layer.trainable = False
#    for layer in model.layers[172:]:
#       layer.trainable = True
#    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
#    # Depois que carrega a rede ...
#    model.fit(training_images,training_labels, batch_size=16, epochs=50,validation_data=(testing_images,testing_labels))
#    #model.fit_generator(datagen.flow(flow_training.x,flow_training.y),steps_per_epoch =16,epochs= 50,show_accuracy=True,validation_data=(datagen.flow(flow_testing.x, flow_testing.y)))
#    #fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
#    model.save_weights("fine_tune_inception.h5w")
