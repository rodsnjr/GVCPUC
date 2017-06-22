#!-*- conding: utf8 -*-
" Cliente para anotar diretórios de imagens com nossas labels "
from gvc.general.application.image_ctrl import *
from gvc.general.application.image_view import *
import argparse
from skimage import io
# Rodar o cliente que lê um diretório e guarda o CSV do Dataset (deste diretório)
# Teclas:
# 1, 2,3 (labels)
# Q - Save and Close
# S - Save
# N, P (Next/Previous)

def parse():
    " Parse the args to launch this app "
    parser = argparse.ArgumentParser(description='Read images from and save the labels on csv.')

    parser.add_argument('--dir', type=str, help='Path of the images')
    # parser.add_argument('--output', type=str, help="Path of the csv")

    args = parser.parse_args()
    return args

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

if __name__ == "__main__":
    args = parse()
    # directory = args.dir
    directory = "D:/Datasets/testing"
    files = listdir_fullpath(directory)
    view = ImageUI(default_size=(300, 300))
    classifier = ClassifierCtrl()
    ctrl = ImageController(view=view, files=files, classifier=classifier)
    ctrl.start()