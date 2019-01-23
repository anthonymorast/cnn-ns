import os
import csv

if __name__ == '__main__':
    test_dir = 'test_bw'
    train_dir = 'train_bw'

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

    with open('train_bw.csv', 'w') as f:
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

    with open('test_bw.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
