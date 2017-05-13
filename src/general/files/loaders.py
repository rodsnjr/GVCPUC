import numpy
import resources

#    Hidden Default Functions
def default_labels_parser(labels):
    binary_labels = []
    for label in labels:
        binary_labels.append(default_label_parser(label))
    return binary_labels

def default_label_parser(label):
    if label == 'door':
        return 1
    elif label == 'indoors':
        return 2

def default_hog(image):
    from skimage.feature import hog
    # Return features
    return hog(image, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), feature_vector=True)

class Flow:
    def __init__(self):
        self.x, self.y = [], []
        self.numpyX, self.numpyY = None, None

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)
    
    def X(self):
        if self.numpyX is None:
            self.numpyX = numpy.array(self.x)
        return self.numpyX

    def Y (self):
        if self.numpyY is None:
            self.numpyY = numpy.array(self.y)
        return self.numpyY

class ImageLoader:
    def __init__(self, path=resources.LOADER_GVCPDIR, dataset=None, size=(100, 100), channels='gray'):
        if dataset is not None:
            self.path = dataset.path

        self.size = size
        self.channels = channels
    
    def pre_process(self, image):
        from skimage import color, transform
        proc = image
        
        if self.channels=='gray':
            proc = color.rgb2gray(image)
        
        img = transform.resize(proc, self.size)

        return img
    
    def __crop(self, image):
        from skimage import color
        image = color.rgb2gray(image)
        w, h = image.shape
        cropW, cropH = w // 3, h // 3
        # Tira o primeiro pedaco da imagem
        x, y, x1, y1 = 0, 0, cropW, cropH
        left = image[x:x1, y:y1]
        x2, y2, x3, y3 = cropW * 2, cropH * 2, w, h
        center = image[x1:x2, y1:y2]
        right = image[x2:x3, y2:y3]
        return left, center, right

    def __parse_csv_line(self, line):
        from skimage import io, transform    
        image = io.imread(self.path + line[0])
        if len(line) > 2:
            labels = (line[1], line[2], line[3])
        else:
            labels = line[1]
        return image, labels

    def crop_flow_from_csv(self, file, lparser=default_labels_parser):
        import csv
        
        flowLoader = Flow()

        with open(self.path + file, newline='') as csvfile:
            lines = csv.reader(csvfile)
            for line in lines:
                img, labels = self.__parse_csv_line(line)
                blabels = lparser(labels)
                for crop, label in zip(self.__crop(img), blabels):
                    flowLoader.add(self.pre_process(crop), label)
        
        return flowLoader
    
    def flow_from_csv(self, file, lparser=default_label_parser):
        import csv

        flowLoader = Flow()

        with open(self.path + file, newline='') as csvfile:
            lines = csv.reader(csvfile)
            for line in lines:
                img, label = self.__parse_csv_line(line)
                blabel = lparser(label)
                flowLoader.add(self.pre_process(img), blabel)
        
        return flowLoader
   
    def flow_from_directory(self, classes):
        from skimage import io

        flowLoader = Flow()

        for index, directory in enumerate(classes):
            images = io.imread_collection(self.path + directory)
            for image, fileName in zip(images, images.files):
                flowLoader.add(self.pre_process(image), index)
        
        return flowLoader


class FeatureLoader(ImageLoader):
    def __init__(self, path=resources.LOADER_GVCPDIR, dataset=None, size=(100, 100), channels='gray',  features = default_hog):
        super().__init__(path, dataset=dataset, size=size, channels=channels)
        self.features = features
    
    def pre_process(self, image):
        img = super().pre_process(image)
        return self.features(img)