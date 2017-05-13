import os
import numpy
dir = os.path.dirname(__file__)

doors_path = "C:/Imagens Treinamento/Cropped_Doors/"
indoors_path = "C:/Imagens Treinamento/Indoors/"
test_path = "C:/Imagens Treinamento/Test/"

def get_images_labels_path(paths, labels):
    from skimage import io, transform
    img_array = []
    labels = []
    
    for path, label in zip(paths, labels):
        images = io.imread_collection(path)
        for image, fileName in zip(images, images.files):
            labels.append(label)
            resized = transform.resize(image, (100, 100))
            try:
                img_array.append(numpy.reshape(resized, (100, 100, 3)))
            except ValueError:
                print("Img Shape error")

    return (img_array, labels)


def get_images_path(path):
    imgexit = []
    images = io.imread_collection(path)
    for image, fileName in zip(images, images.files):
        imgexit.append(image)
    return imgexit


def get_classifier_train_images():
    """
    Get all the images for training
    - returns the images and labels
    """
    images, labels = get_images_labels_path((doors_path + "*.jpg", indoors_path + "*.jpg"), (1,2))
    
    x_train = numpy.array(images)
    y_train = numpy.array(labels)
    
    return x_train, y_train


def get_classifier_test_images():
    """Get all the images for testing
    - returns the images and labels
    """
    images, labels = get_images_labels_path((test_path + "Doors/*.jpg", test_path + "Indoors/*.jpg"), (1, 2))
    
    x_train = numpy.array(images)
    y_train = numpy.array(labels)
    
    return x_train, y_train