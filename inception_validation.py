import resources
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2

import tensorflow as tf

from gvc.general.timer import Timer
from difflib import SequenceMatcher

def load_imgs(path = "D:/Datasets/dataset_com_csv", file_name = "dataset_door.csv", res = 300, chennels = 3):
    #print file_name
    file = open(os.path.join(path, file_name) ,"r")
    lines = file.readlines()
    file.close()

    images = []
    labels = []
    #input de dados para o treinamento. 
    for x in range(0, 200): 
        buffer_str = lines[x].split(',')
        image_path = path + buffer_str[0]
        # image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        
        img = cv2.imread(image_path)
        
        if img is None:
            print("no image {}{}".format(path, buffer_str[0]))
            continue
               
        height, width, channels = img.shape
        
        #separa as imagens e coloca uma em casa posicao..
        left = img[0:height,0:width//3]
        left = cv2.resize(left, (res, res)) 
            
        center = img[0:height,width//3:2*width//3]
        center = cv2.resize(center, (res, res))

        right = img[0:height,2*width//3:width]
        right = cv2.resize(right, (res, res))

        img = cv2.resize(img, (res, res))

        images.append({ 'left':left, 'center':center, 'right':right})
        labels.append({'left' : buffer_str[1], 'center': buffer_str[2], 'right': buffer_str[3]})
    
    return images, labels

def show(left, center, right):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(8, 2.5),
                                    sharex=True,
                                    sharey=True)
    ax0.imshow(left)
    ax0.set_title('left')
    ax0.axis('off')
    ax0.set_adjustable('box-forced')

    ax1.imshow(center)
    ax1.set_title('center')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')

    ax2.imshow(right)
    ax2.set_title('right')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')


def load_graph():
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.FastGFile(resources.CLASSIFIERS+'/graph.pb', "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

def similar(a, b):
    if a == 'indoor' and b == 'indoors':
        return True
    elif a == 'doors' and ('door' in b and 'in' not in b):
        return True
    elif a == 'stairs' and b == 'stair':
        return True
    return False


def get_predictions(predictions):
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        return (human_string, score)

if __name__ == "__main__":

    timer = Timer()
    x, y = load_imgs(res=224)

    graph = load_graph()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(resources.RESOURCES+"labels.txt")]
    
    total = 0
    acertos = 0

    with tf.Session(graph=graph) as sess:
        softmax = sess.graph.get_tensor_by_name('prefix/final_result:0')

        for image_dict, label_dict in zip(x, y):
            timer.tic()

            for key, value in image_dict.items():
                total+=1
                img = np.expand_dims(value, axis=0)
                predictions = sess.run(softmax, {'prefix/DecodeJpeg:0' : img[0]})
                string, score = get_predictions(predictions)
                acertos += 1 if similar(string,label_dict[key]) else 0
                print(key,' Predicted {} with score {}'.format(string, str(score)))

            print("Total Prediction Time: ", timer.toc())
    
    print("Acertos - {}, no total de {}".format(acertos, total))