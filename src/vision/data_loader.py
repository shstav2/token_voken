import logging

import numpy as np
import PIL
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from src.common.debug import one_percent_chance
from src.common.display_utils import WRN


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

TAG = '[ImageFolderWithPaths]'

IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

EMPTY_IMG = np.nan
is_empty = lambda img: np.isnan(img).any().item()

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    # fixed_image_standardization
])

def get_tensor_image(img_path):
    pil_img = default_loader(img_path)
    img_tensor = IMG_TRANSFORM(pil_img)
    return img_tensor

# Taken from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Returns: tuple of 3 items:
            [0] - Image as a torch tensor (e.g of shape [3, 224, 224])
            [1] - Label (folder name) - 364
            [2] - Image path (e.g: /home/stav/Data/PATS_DATA/Videos/oliver/7y1xJAVZxXg/104815/FacesAll/00364/face_0.jpg)
        """
        # The image file path
        path = self.imgs[index][0]
        # This is what ImageFolder normally returns
        try:
            tensor_img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        except PIL.UnidentifiedImageError:
            # account for empty image
            tensor_img = EMPTY_IMG
            label = self.imgs[index][1]

        # Add image path to dataset item
        tuple_with_path = (tensor_img, label, path)

        _debug_log(index, tuple_with_path)
        return tuple_with_path


def is_face_path(path):
    return 'face_' in path


def get_data_loader(data_dir):
    dataset = ImageFolderWithPaths(data_dir, transform=trans, is_valid_file=is_face_path)
    dataloader = DataLoader(dataset)
    logger.info(f'{TAG} Initialize dataloader using dataset: {dataset}')
    return dataloader


def _debug_log(index, tuple_with_path):
    img, label, path = tuple_with_path

    is_img_empty = is_empty(img)
    if is_img_empty: logger.warning(f'{TAG} {WRN}  Empty image: {path}')

    should_debug = is_img_empty or one_percent_chance()
    if should_debug:
        logger.info(f'{TAG}.__getitem__(index={index})'
                    f'\n\t[0] -Image={img.shape if not is_img_empty else None}'
                    f'\n\t[1] -Label={label}'
                    f'\n\t[2] -Path={path}')

