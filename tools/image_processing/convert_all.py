from PIL import Image
import sys
import os
from ppm_crop import crop

if __name__ == '__main__':
    indir = os.path.join('../../', 'ansys')
    outdir = os.path.join('.', 'data')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    count = 0
    for filename in os.listdir(indir):
        if os.path.isdir(os.path.join(indir,filename)) and filename.startswith('re'):
            path = os.path.join(indir, filename, 'vel_magnitude', 'output')
            for f in os.listdir(path):
                if f.endswith('.ppm'):
                    img_out_path = filename+'_'+str(int(f[-8:-4])*10)+'.png'
                    img_in_path = os.path.join(path, f)
                    print(img_in_path)
                    img = Image.open(img_in_path)
                    img = crop(img)
                    count += 1
                    img.save(os.path.join(outdir, img_out_path))

    print(count, "files converted and saved in", outdir)
