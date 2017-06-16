from gvc.general.files.loaders import ImageLoader
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import resources
import numpy as np

def local_loader():
    loader = ImageLoader('D:/Datasets/gvc_dataset/', size=(224, 224), channels='rgb')
    flow = loader.crop_flow_from_csv("door_e_indoor_testing.csv")
    return flow

if __name__ == "__main__":
    test_flow = local_loader()
    
    from keras import backend as K

    model = ResNet50(weights='imagenet')
    # incp = InceptionV3(weights='imagenet', input_shape=(224, 224, 3))
    x = test_flow.X()
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    preds = model.predict(x)
    # incp_preds = incp.predict(x)
    print("Resnet")
    # print('Predicted: ', preds)
    print('Decoded Predicted:', decode_predictions(pred, top=3))
    print('\n\n\n')
    # print('Inception', decode_predictions(incp_preds, top=3)[1])

