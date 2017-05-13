"""
    Testing a Simple SVM with HoG Features
"""
from gvc.general.files.loaders import FeatureLoader
from gvc.classifiers.model import SVC
from gvc.classifiers.bow import BOW

# Tá na minha máquina na minha casa ...
def local_loader():
    loader = FeatureLoader(
        path='C:/Imagens Treinamento/'
    )

    train_flow = loader.crop_flow_from_csv('annotation.csv')
    test_flow = loader.crop_flow_from_csv('test.csv')

    return train_flow, test_flow

if __name__ == "__main__":
    
    train_flow, test_flow = local_loader()

    # Simple SVM
    mySvm = SVC()
    mySvm.train_flow(train_flow)
    print("SVC HoG Evaluation {}".format(mySvm.evaluate_flow(test_flow)))

    # Simple BoW
    bow = BOW()
    bow.train_flow(train_flow)
    print("BoW w/ SVC HoG Evaluation {}".format(bow.evaluate_flow(test_flow)))