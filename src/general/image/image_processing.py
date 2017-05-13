from skimage.feature import hog
from skimage import data, color, exposure, transform
from outputs import visualize_sliding

def extract_features_images(images):
    "Extract local features of the images using HOG Descriptors"
    features = []
    for image in images:
        features.append(extract_features(image))
    return features


def extract_features(image):
    gray = color.rgb2gray(image)
    resized = transform.resize(gray, (50, 50))
    hog = extract_hog(resized)
    return hog


def extract_hog(image):
    fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), feature_vector=True)
    return fd


def crop_image(image):
    gray = color.rgb2gray(image)
    w, h = gray.shape
    cropW, cropH = w // 3, h // 3
    # Tira o primeiro pedaco da imagem
    x, y, x1, y1 = 0, 0, cropW, cropH
    left = gray[x:x1, y:y1]
    x2, y2, x3, y3 = cropW * 2, cropH * 2, w, h
    center = gray[x1:x2, y1:y2]
    right = gray[x2:x3, y2:y3]
    return left, center, right

class Sliding_Window():
    def __init__(self, classifier, min_wdw_sz=(100,40), step_size=(10,10)):
        self.classifier = classifier 
        self.min_wdw_sz, self.step_size= min_wdw_sz, step_size
        self.detections, self.cd= [], []

    def slide(self, image, window_size, step_size):
        for y in range(0, image.shape[0], step_size[1]):
            for x in range(0, image.shape[1], step_size[0]):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
    
    def slide_window(self, image, visualize):        
        for (x, y, im_window) in self.slide(image, self.min_wdw_sz, self.step_size):
            if im_window.shape[0] != self.min_wdw_sz[1] or im_window.shape[1] != self.min_wdw_sz[0]:
                continue
            # Calculate the HOG features
            fd = extract_features(im_window)
            pred = self.classifier.predict(fd)
            if pred == 1:
                df = self.classifier.decision_function(fd)
                self.detections.append((x, y, df, int(
                    self.min_wdw_sz[0] * (self.downscale**self.scale)), 
                    int(self.min_wdw_sz[1] * (self.downscale**self.scale))))
                self.cd.append(self.detections[-1])
            if visualize:
                visualize_sliding(image, im_window, (x, y), self.cd)

    def sliding_window(self, image, do_nms=True, visualize=False):
        self.scale, self.downscale=0, 1
        self.detections, self.cd= [], []
        self.slide_window(image, visualize)
        
        if nms:
            from nms import nms
            self.detections = nms(self.detections)
        return self.detections
    
    def sliding_window_downscale(self, image, do_nms=True, downscale=1.25, visualize=False):
        from skimage.transform import pyramid_gaussian
        self.scale, self.downscale=0, downscale
        self.detections=[]
        for im_scaled in pyramid_gaussian(image, downscale=self.downscale):
            self.cd = []
            if im_scaled.shape[0] < self.min_wdw_sz[1] or im_scaled.shape[1] < self.min_wdw_sz[0]:
                break
            self.slide_window(im_scaled, visualize)
            self.scale += 1
        
        if do_nms:
            from nms import nms
            self.detections = nms(self.detections)
        return self.detections