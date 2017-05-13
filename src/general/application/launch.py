from .csv_model import CSVModel
from .image_ctrl import ImageController
from .image_view import ImageUI

from skimage import io

def launch(directory, csvwrite):
    images = io.imread_collection(directory)
    view = ImageUI()
    model = CSVModel(csvwrite)
    ctrl = ImageController(view=view, model=model, files=images.files)
    ctrl.start()