from PIL import Image

def crop(image):
    # box = (315, 147, 851, 415) ## whole cylinder
    box = (350, 200, 700, 360)  ## mostly just wake and cylinder
    box = (285, 160, 635, 320)  ## broken 150s
    return image.crop(box)

if __name__ == '__main__':
    # image = Image.open('animation-2_0001.ppm')
    # image.save('test.png')
    img = Image.open('re150.png')
    img = crop(img)
    img.save('crop.png')
