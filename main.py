"""
    Testing a Simple SVM with HoG Features
"""
from gvc.general.files.loaders import FeatureLoader
from gvc.classifiers.model import SVC
from gvc.classifiers.bow import BOW

def local_loader():
    # TODO - tem que colocar um args parser pra esses caras ...
    loader = FeatureLoader('D:/Datasets/gvc_dataset/')

    train_flow = loader.crop_flow_from_csvs(['dataset_door.csv','dataset_indoor.csv', 'dataset_stairs.csv'])
    test_flow = train_flow.extract_flow(1000)
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