from PIL import Image

def crop(image):
    box = (315, 147, 851, 415)
    return image.crop(box)
