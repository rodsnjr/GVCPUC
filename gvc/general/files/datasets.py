"TODO - Tem que organizar isso aqui, aí até fica legal!"
from keras.utils import get_file
from enum import Enum

GVCPUC = "GVC_PUC"

class Resources(Enum):
    GVC_URL = "https://github.com/rodsnjr/gvc_dataset/archive/master.zip"

class Dataset:
    def __init__(self, path, validation_path, test_path, classes):
        self.path = path
        self.classes = classes
        self.validation_path = validation_path
        self.test_path = test_path

class CSVDataset:
    def __init__(self, path, dataset_csv, validation_csv, test_csv):
        self.path = path
        self.dataset_csv = dataset_csv
        self.validation_csv = validation_csv
        self.test_csv = test_csv


def __download(dataset):
    if dataset == GVCPUC:
        return get_file(resources.RES_DATASETS_PATH + "GVCP.zip", resources.URL_GVCPUC)

def __extract(dataset, file):
    import zipfile
    zip_ref = zipfile.ZipFile(file, 'r')
    zip_ref.extractall(resources.RES_DATASETS_PATH + "/" + dataset)
    zip_ref.close()
    return resources.RES_DATASETS_PATH + "/" + dataset

def __exists(dataset):
    return True

def __cache(dataset):
    return resources.RES_DATASETS_PATH + "/" + dataset

def load_dataset(dataset=GVCPUC):
    # Adiciona o switch com os metodos de LOAD
    # Pra retornar um objeto Dataset
    if __exists(dataset) is False:
        download = __download(dataset)
        path = __extract(dataset, download)
    else:
        path = __cache(dataset)
    
    if dataset == GVCPUC:
        return __load_gvcds(path)

def load_csv_dataset(dataset=GVCPUC):
    # Adiciona o switch com os metodos de LOAD
    # Pra retornar um objeto Dataset
    download = __download(dataset)
    path = __extract(dataset, download)
    if dataset == GVCPUC:
        return __load_csv_gvcds(path)