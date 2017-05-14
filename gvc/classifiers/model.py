from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib


# Validations
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



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
    
    def cross_val(self, x, y):
        predicted = cross_val_predict(self.classifier, x, y, cv=10)
        return metrics.accuracy_score(y, predicted)

class SVC(Model):
    def __init__(self):
        from sklearn import svm
        super().__init__(svm.SVC())