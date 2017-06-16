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

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Input
import argparse



def parse():
    " Parse the args to launch this app "
    parser = argparse.ArgumentParser(description='Read images from and save the labels on csv.')
    
    parser.add_argument('--dir', type=str, help='Path of the images')
    parser.add_argument('--training', type=str, help="Name of the training csv")
    parser.add_argument('--testing', type=str, help="Name of the testing csv")
    
    args = parser.parse_args()
    return args

def local_loader(path_name,filename):
    # TODO - tem que colocar um args parser pra esses caras ...
    loader = ImageLoader(path_name)

    flow = loader.crop_flow_from_csv(filename)
    # test_flow = train_flow.extract_flow(1000)
    return flow

if __name__ == "__main__":

    args = parse()

    flow_training= local_loader(args.dir,args.training)
    flow_testing = local_loader(args.dir,args.testing)
 
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )


    input_tensor = Input(shape=(300, 300, 3))  # this assumes K.image_data_format() == 'channels_last'

    base_model = InceptionV3(input_tensor=input_tensor,weights='imagenet', include_top=True)
   
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    #modelo a ser treinado
    model = Model(inputs=base_model.input, outputs=predictions)
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    for layer in model.layers[:172]:
       layer.trainable = False
    for layer in model.layers[172:]:
       layer.trainable = True
    
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

#    print base_model.summary()
#
#    for i, layer in enumerate(base_model.layers):
#        print(i, layer.name)
    #model = build_model()
    
    # Depois que carrega a rede ...
    model.fit_generator(datagen.flow(flow_training.x(), flow_training.y(), batch_size=16),steps_per_epoch=len(flow.x()), epochs=50,show_accuracy=True,validation_data=(datagen.flow(flow_testing.x(), flow_testing.y())))
    model.save_weights("fine_tune_inception.h5w")
