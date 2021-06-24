from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision import datasets

IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

def get_tensor_image(img_path):
    pil_img = default_loader(img_path)
    img_tensor = IMG_TRANSFORM(pil_img)
    return img_tensor

# Taken from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        print(f'ImageFolderWithPaths.__getitem__(index={index})\n\t[0] - {original_tuple[0].shape}\n\t[1] - {original_tuple[1]},\n\t[2] - {path}')
        return tuple_with_path
