"""
    Testing a Simple SVM with HoG Features
"""
from gvc.general.files.loaders import FeatureLoader
from gvc.classifiers.model import SVC
from gvc.classifiers.bow import BOW

def local_loader():
    # TODO - tem que colocar um args parser pra esses caras ...
    loader = FeatureLoader('D:/Datasets/gvc_dataset/')

    flow = loader.crop_flow_from_csvs(['dataset_door.csv','dataset_indoor.csv', 'dataset_stairs.csv'])
    # test_flow = train_flow.extract_flow(1000)
    return flow

if __name__ == "__main__":

    train_flow = local_loader()

    # Simple SVM
    mySvm = SVC()
    mySvm.train_flow(train_flow)

    # Simple BoW
    bow = BOW()
    bow.train_flow(train_flow)
    svc_validate = mySvm.cross_val(train_flow.X(), train_flow.Y())
    bow_validate = bow.cross_val(train_flow.X(), train_flow.Y())

    print("SVC HoG Cross Val Score {}".format(svc_validate))
    print("BoW w/ SVC HoG Cross Val Score {}".format(bow_validate))

    