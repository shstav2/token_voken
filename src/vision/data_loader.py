import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])


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
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        print(f'ImageFolderWithPaths.__getitem__(index={index})\n\t[0] - {original_tuple[0].shape}\n\t[1] - {original_tuple[1]},\n\t[2] - {path}')
        return tuple_with_path

def is_face_path(path):
    return 'face_' in path

def get_data_loader(data_dir):
    dataset = ImageFolderWithPaths(data_dir, transform=trans, is_valid_file=is_face_path)
    dataloader = DataLoader(dataset)
    return dataloader
