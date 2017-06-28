from .csv_model import CSVModel
from .image_ctrl import ImageController
from .image_view import ImageUI

import os

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def launch(directory, csvwrite):
    images = listdir_fullpath(directory)
    view = ImageUI(default_size=(300, 300))
    model = CSVModel(csvwrite)
    ctrl = ImageController(view=view, model=model, files=images)
    ctrl.start()