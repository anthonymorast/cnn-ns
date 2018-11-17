import os
import csv

if __name__ == '__main__':
<<<<<<< HEAD
    test_dir = 'test_bw'
    train_dir = 'train_bw'
=======
    test_dir = 'test'
    train_dir = 'train'
>>>>>>> 92ad549a63fd9083d85972a6812eacf9cb68cdd5

    train_files = os.listdir(os.path.join('.', train_dir))
    test_files = os.listdir(os.path.join('.', test_dir))

    rows = [['path', 'timestep', 're']]
    for f in train_files:
        path = os.path.join(train_dir, f)
        re = f[2:]
        re = re[:re.index('_')]
        re = int(re)
        timestep = f[(f.index('_')+1):-4]
        rows.append([path, timestep, re])

<<<<<<< HEAD
    with open('train_bw.csv', 'w') as f:
=======
    with open('train.csv', 'w') as f:
>>>>>>> 92ad549a63fd9083d85972a6812eacf9cb68cdd5
        writer = csv.writer(f)
        writer.writerows(rows)

    rows = [['path', 'timestep', 're']]
    for f in test_files:
        path = os.path.join(test_dir, f)
        re = f[2:]
        re = re[:re.index('_')]
        re = int(re)
        timestep = f[(f.index('_')+1):-4]
        rows.append([path, timestep, re])

<<<<<<< HEAD
    with open('test_bw.csv', 'w') as f:
=======
    with open('test.csv', 'w') as f:
>>>>>>> 92ad549a63fd9083d85972a6812eacf9cb68cdd5
        writer = csv.writer(f)
        writer.writerows(rows)
