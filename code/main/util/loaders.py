from PIL import Image
from torchvision import get_image_backend
#import accimage


def pil_loader(path):
    """  
    load image functions taken from official documentation of pytorch
    https://github.com/pytorch/vision/blob/4ec38d496db69833eb0a6f144ebbd6f751cd3912/torchvision/datasets/folder.py#L57
    """
    # open path as file to avoid
    # ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB') # to use pytorch pretrained models we need RGB

def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
  