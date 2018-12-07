from PIL import Image
import sys
import os

if __name__ == '__main__':
    dirs = ['test', 'train']

    for dir in dirs:
        print('Processing directory:', dir)
        outdir = dir+'_bw'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for filename in os.listdir(dir):
            print('\tFile:', os.path.join(dir, filename))
            img = Image.open(os.path.join(dir, filename))
            img = img.convert('L')
            img.save(os.path.join(outdir, filename))
