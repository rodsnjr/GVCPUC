from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib

class Model():
    def __init__(self, classifier):
        self.classifier = classifier
    
    def train(self, x, y):
        self.classifier.fit(x, y)
    
    def train_flow(self, flow):
        self.classifier.fit(flow.X(), flow.Y())
    
    def predict(self, x):
        return self.classifier.predict(x)
    
    def save(self, path):
        joblib.dump(self.classifier, path)
    
    def load(self, path):
        self.classifier = joblib.load(path)
    
    def evaluate(self, x, y):
        return self.classifier.score(x,y)

    def evaluate_flow(self, flow):
        return self.classifier.score(flow.X(), flow.Y())

class SVC(Model):
    def __init__(self):
        from sklearn import svm
        super().__init__(svm.SVC())