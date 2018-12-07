from PIL import Image
import sys
import os
from ppm_crop import crop


def usage():
    print("python3 ppm_to_png.py <input_dir> <output_dir> \{OPTIONAL: crop=[0,1]\}")
    print("\tcrop - if non-zero crop the images, defaults to true (1)")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        exit(0)

    c = 1
    if len(sys.argv) > 3:
        c = int(sys.argv[3])

    indir = sys.argv[1]
    outdir = sys.argv[2]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    count = 0
    for filename in os.listdir(indir):
        if filename.endswith('.ppm'):
            count += 1
            image = Image.open(indir+filename)
            step = int(filename[-8:-4])*10
            if c:
                image = crop(image)
            image.save(outdir+'re150_'+str(step)+'.png')

    print("Converted", count, "ppms to pngs.")
