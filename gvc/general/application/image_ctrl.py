import cv2
import tensorflow as tf
import numpy as np
import resources

class ImageController:
    def __init__(self, view=None, files=None, model=None, classifier=None):
        self.model = model
        self.files = files
        self.view = view
        self.current_file = 0
        self.classifier = classifier
    
    # Load the current progress from the model file
    def load_progress(self):
        if self.model is not None:
            self.model.load()
            self.current_file = len(self.model.items)

    # Start the Window/Controller Loop
    def start(self):
        self.load_progress()
        self.view.load_image(self.files[self.current_file])
        self.view.show()

        while True:
            key = cv2.waitKey(1) & 0xFF
            # next image
            if key == ord("n"):
                self.next()
            # previus image
            elif key == ord("p"):
                self.previous()
            # Labels
            elif key == ord("1"):
                self.label1()
            elif key == ord("2"):
                self.label2()
            elif key == ord("3"):
                self.label3()
            elif key == ord('s'):
                self.save_current()
            # quit
            elif key == ord('c'):
                self.classify()
            elif key == ord("q"):
                break
        self.close()

    # Directory Travelling Events
    def next(self):
        if self.model is not None:
            self.model.add(self.files[self.current_file], self.view.shown.describe())
            self.model.save()
        
        self.current_file += 1
        self.view.show_next(self.files[self.current_file])
        # if self.on_next:
        #    self.on_next()
    
    def save_current(self):
        if self.model is not None:
            self.model.add(self.files[self.current_file], self.view.shown.describe())
            self.model.save()

    def previous(self):
        self.current_file -= 1
        self.view.show_item(self.model.last())
    
    def close(self):
        if self.model is not None:
            self.model.save()
        
    # GUI Drawing / Changing Events
    def label1(self):
        self.view.shown.change_label('left')
        self.view.refresh()
    
    def label2(self):
        self.view.shown.change_label('center')
        self.view.refresh()
    
    def label3(self):
        self.view.shown.change_label('right')
        self.view.refresh()
    
    def classify(self):
        label_dict = self.classifier.run(self.view.cropped_images(self.classifier.resolution))
        self.view.shown.change_labels(label_dict)
        self.view.refresh()
    
    # Controller Event Setters
    def on_next(self, on_next):
        self.on_next = on_next
    
    def on_previous(self, on_previous):
        self.on_previous = on_previous
    
    def on_close(self, on_close):
        self.on_close = on_close

class ClassifierCtrl:
    def __init__(self, resolution=224):
        self.resolution = resolution
        self.graph = self.load_graph()
        self.label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(resources.RESOURCES+"doors.txt")]
        
    def load_graph(self):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.FastGFile(resources.CLASSIFIERS+'/doors.pb', "rb") as f:
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

    def run(self, image_dict):
        classification = {}
        with tf.Session(graph=self.graph) as sess:
            softmax = sess.graph.get_tensor_by_name('prefix/final_result:0')
            for key, value in image_dict.items():
                img = np.expand_dims(value, axis=0)
                predictions = sess.run(softmax, {'prefix/DecodeJpeg:0' : img[0]})
                classification[key] = self.get_predictions(predictions)
        return classification
    
    def get_predictions(self, predictions):
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = self.label_lines[node_id]
            score = predictions[0][node_id]
            return human_string