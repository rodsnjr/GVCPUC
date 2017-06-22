# Interface gráfica para carregar imagens de um diretório 
# mostrar, e alterar as labels em cada quadrante
import cv2
import os
import numpy as np

class ImageUI:
    def __init__(self, default_size=(150, 150)):
        self.title = "Imagem - {}"
        self.default_size = default_size
        self.image = None
        self.resized_image = None
        self.gray_image = None

    def load_image(self, filename):
        self.title = self.title.format(filename)
        self.image = cv2.imread(filename)        
        self.resized_image = cv2.resize(self.image, self.default_size,
            interpolation = cv2.INTER_CUBIC)
        # self.gray_image = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2GRAY)
        
    def show(self):
        self.shown = ImageShow(self.resized_image)
        self.shown.redraw()
        cv2.imshow(self.title, self.shown.image)
    
    def show_item(self, item):
        cv2.destroyWindow(self.title)
        self.title = "Imagem - {}"
        self.load_image(item[0])
        self.shown = ImageShow(self.resized_image)
        self.shown.build_labels_item(item)
        self.shown.redraw()
        cv2.imshow(self.title, self.shown.image)
        
    def show_next(self, filename):
        cv2.destroyWindow(self.title)
        self.title = "Imagem - {}"
        self.load_image(filename)
        self.show()
    
    def refresh(self):
        cv2.imshow(self.title, self.shown.image)
    
    def cropped_images(self, resolution):
        height, width, channels = self.image.shape

        left = self.image[0:height,0:width//3]
        left = cv2.resize(left, (resolution, resolution)) 
            
        center = self.image[0:height,width//3:2*width//3]
        center = cv2.resize(center, (resolution, resolution))

        right = self.image[0:height,2*width//3:width]
        right = cv2.resize(right, (resolution, resolution))

        return {'left':left, 'center':center, 'right':right}


class ImageShow: 
    def __init__(self, image):
        self.oimage = image.copy()
        self.image = image.copy()
        self.three_crop()
        self.build_labels()
    
    def build_labels(self):
        self.labels = { 'left' : Label((10, self.h - 10)), 
            'right' : Label((self.x2 + 10, self.h - 10)), 
            'center' : Label((self.x1 + 10, self.h - 10)) 
        }
    
    def build_labels_item(self, item):
        self.build_labels()
        self.labels['left'].label = item[1]
        self.labels['center'].label = item[2]
        self.labels['right'].label = item[3]
    
    # Return a string of the description of the labels
    def labels_to_string(self):
        left = self.labels['left']
        center = self.labels['center']
        right = self.labels['right']
        description = "{},{},{}".format(left.label, center.label, right.label)
        return description
    
    # Return the description of the labels
    def describe(self):
        left = self.labels['left'].label
        center = self.labels['center'].label
        right = self.labels['right'].label
        return left, center, right
    
    def redraw(self):
        self.image = self.oimage.copy()
        self.draw_lines()
        self.draw_labels()
    
    def change_label(self, position):
        self.labels[position].next()
        self.redraw()
    
    def change_labels(self, labels):
        for key, val in labels.items():
            self.labels[key].change(val)
        self.redraw()

    def three_crop(self):
        self.w, self.h = self.image.shape[0], self.image.shape[1]
        self.x1 = self.w // 3
        self.x2 = self.x1 * 2
    
    def draw_lines(self):
        cv2.line(self.image, (self.x1, 0), (self.x1, self.h), (255,0,0), 5)
        cv2.line(self.image, (self.x2, 0), (self.x2, self.h), (255,0,0), 5)

    def draw_labels(self):
        for k, v in self.labels.items():
            cv2.putText(self.image, v.label, v.xy, Label.FONT, 0.3, (255,0,0), 1, cv2.LINE_AA)

class Label:
    LABELS = ['door', 'indoors', 'stairs']
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, xy):
        self.label = 'door'
        self.xy = xy
        self.current = 0
    
    def next(self):
        self.current = self.current+1 if self.current < (len(Label.LABELS)-1) else 0
        self.label = Label.LABELS[self.current]
    
    def change(self, value):
        self.current = translate(value)
        self.label = Label.LABELS[self.current]

def translate(a):
    if a == 'indoor':
        return 1
    elif a == 'doors':
        return 0
    elif a == 'stairs':
        return 2