import os
from random import shuffle
from shutil import copyfile

if __name__ == '__main__':
    data_dir = 'data'
    train_dir = 'train'
    test_dir = 'test'

    files = os.listdir(data_dir)

    shuffle(files)
    shuffle(files)
    shuffle(files)
    shuffle(files)

    train_size = int(len(files)*.8)
    test_size = int(len(files) - train_size)
    train_files = files[0:train_size]
    test_files = files[train_size:len(files)]

    for file in train_files:
        src = os.path.join(data_dir, file)
        dst = os.path.join(train_dir, file)
        copyfile(src, dst)

    for file in test_files:
        src = os.path.join(data_dir, file)
        dst = os.path.join(test_dir, file)
        copyfile(src, dst)

    
