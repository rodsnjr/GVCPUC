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
    labels = np.ndarray(shape = (len(lines)*3,2))

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
                labels[counter_lbl]=[1,0]
            if(buffer_str[i]=="indoors"):
                labels[counter_lbl]=[0,1]
            if(buffer_str[i]=="stairs"):
                labels[counter_lbl]=[0,1]
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


if __name__ == "__main__":

    args = parse()

    
    training_images,training_labels = load_imgs(args.dir,args.training,299,3)
    testing_images,testing_labels = load_imgs(args.dir,args.testing,299,3)

    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')
    training_images /= 255
    testing_images /= 255


    #input = Input(shape=training_images[0].shape ,name = 'image_input')
    #print input
    base_model = InceptionV3(weights='imagenet', include_top=False) # importa o inception com o input acima.
   

    #print base_model.summary()

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x) #eh tipo um fully connected, link: https://www.quora.com/What-is-global-average-pooling
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x) #ultima camada da rede, com duas classes. 
    model = Model(input=base_model.input, output=predictions)
    print model.summary()


    #seta as camadas 172 para cima para serem treinadas. AS outras nao serao modificadas..
    for layer in model.layers[:172]:
       layer.trainable = False
    for layer in model.layers[172:]:
       layer.trainable = True
 
    print training_labels
    print testing_labels


    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    model.fit(training_images,training_labels, batch_size=3, epochs=1,validation_data=(testing_images,testing_labels))
