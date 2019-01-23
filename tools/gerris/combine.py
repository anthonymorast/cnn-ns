from gerris2csv import printcsv
import os
import pandas as pd
import numpy as np
import h5py

if __name__ == '__main__':
    idmin = 21
    idmax = 20

# created the alldata.csv
"""
    yvals = dict() ## id -> locations list [x1, y1, x2, y2]
    yvals[1] = [-2.1, 0, -1.1, 0]
    yvals[2] = [-1.5, 0, -1.3, 0]
    yvals[3] = [-2, 0.5, -2, -0.5]
    yvals[4] = [-1, 0.5, -1, -0.5]
    yvals[5] = [-2, 1.5, -1, -1.5]
    yvals[6] = [-1.5, -1.5, -1, 1.5]
    yvals[7] = [-1, -1.5, -1, 1.5]
    yvals[8] = [-1.5, 0, -1, 0.1]
    yvals[9] = [-2, 0, -1.5, 0.1]
    yvals[10] = [-2, 1, -1.8, 1.1]
    yvals[11] = [-0.2, -1, -0.5, -1.1]
    yvals[13] = [-1.292, 0.644, -0.707, 1.914]
    yvals[14] = [-2.428, -2.128, -1.282, -1.679]
    yvals[15] = [-0.560, 0.045, -0.123, 0.323]
    yvals[16] = [-1.564, -0.708, -0.566, 1.404]
    yvals[17] = [-1.447, 1.541, -0.102, 1.316]
    yvals[18] = [-1.142, -1.075, -1.810, 1.903]
    yvals[19] = [-1.805, 0.050, -1.468, 0.431]
    yvals[20] = [-2.071, -1.788, -0.077, 1.576]
    
    d = os.path.join('.')
    alldata = None
    first = True
    for i in range(idmin, idmax+1):
        # didn't use 12, data was messed up
        if i == 12:
            continue
        
        print("Processing id ", i)
        
        # read in the csv
        filename = os.path.join(d, 'id'+str(i)+'_data.csv')
        df = pd.read_csv(filename)
        
        # append y values to the dataframe's columns
        x1 = yvals[i][0]
        y1 = yvals[i][1]
        x2 = yvals[i][2]
        y2 = yvals[i][3]
        df['x1'] = x1
        df['y1'] = y1
        df['x2'] = x2
        df['y2'] = y2
        
        # appending does not work (memory error) rather write to one csv repeatedly
        if first:
            df.to_csv('alldata.csv', index=False) # write/create csv
            first = False
        else:
            df.to_csv('alldata.csv', mode='a', index=False, header=False) # append to csv
            
""" 

## Created 6 different hdf5 files
## before this read in alldata.csv as 250 row chunks, randomized and split into test.csv and train.csv (similar to 
## what's done here)
"""
i = 0
ycols = ['x1', 'x2', 'y1', 'y2']
for chunk in pd.read_csv('train.csv', chunksize=750):
    print("processing chunk", i, chunk.shape)
    xcols = [col for col in chunk.columns if col not in ycols]
    x = chunk[xcols]
    y = chunk[ycols]
    print(len(list(chunk)), len(list(x)), len(list(y)))
    x.to_hdf('trainx_'+str(i)+'.h5', key='trainx')
    y.to_hdf('trainy_'+str(i)+'.h5', key='trainy')
    del x
    del y
    i += 1
"""

### Used to create trainx.h5 and trainy.h5 from the files created above in 750 row chunks
"""
trainx_files = ['trainx_'+str(i)+'.h5' for i in range(0, 6)]
trainy_files = ['trainy_'+str(i)+'.h5' for i in range(0, 6)]

xdata = None
for f in trainx_files:
    print("processing file", f)
    data = pd.read_hdf(f, key='trainx').as_matrix()
    if xdata is None:
        xdata = data
    else:
        xdata = np.append(xdata, data, axis=0)
print(xdata.shape)
h5f = h5py.File('trainx.h5', 'w')
h5f.create_dataset('trainx', data=xdata)
h5f.close()
del xdata
    
ydata = None
for f in trainy_files:
    print("processing file", f)
    data = pd.read_hdf(f, key='trainy').as_matrix()
    if ydata is None:
        ydata = data
    else:
        ydata = np.append(ydata, data, axis=0)
print(ydata.shape)
h5f = h5py.File('trainy.h5', 'w')
h5f.create_dataset('trainy', data=ydata)
h5f.close()
"""

### need to re-shuffle the training dataset
## this was used to craete the traindata.h5 file containg all training data that has been reshuffled
"""
hf = h5py.File('trainx.h5', 'r')
trainx = hf.get('trainx')[()]

hf = h5py.File('trainy.h5', 'r')
trainy = hf.get('trainy')[()]
alldata = np.append(trainx, trainy, axis=1)

print(alldata.shape) # should be 3815 x 281254
np.random.shuffle(alldata)
print(alldata.shape)

h5f = h5py.File('traindata.h5', 'w')
h5f.create_dataset('all', data=alldata)
"""

"""
## Used to create the test dataset
data = pd.read_csv('test.csv')

# randomize the dataset
print('shuffle 1')
data = data.sample(frac=1).reset_index(drop=True)
print('shuffle 2')
data = data.sample(frac=1).reset_index(drop=True)
print('shuffle 3')
data = data.sample(frac=1).reset_index(drop=True)
print('shuffle 4')
data = data.sample(frac=1).reset_index(drop=True)
print(data.shape)

ycols = ['x1', 'x2', 'y1', 'y2']
xcols = [col for col in list(data) if col not in ycols]

xdata = data[xcols]
ydata = data[ycols]

del data

xdata = xdata.values
ydata = ydata.values

print(xdata.shape, ydata.shape)

data = np.append(xdata, ydata, axis=1)
print(data[1:5, range(281250, 281254)])

h5f = h5py.File('testdata.h5', 'w')
h5f.create_dataset('all', data=data)
"""

## size test
hf = h5py.File('../../ml_algs/object_detection/traindata.h5', 'r')
data = hf.get('all')

print(data.shape)

print(data[25, range(281250, 281254)]) ### NOTE: IMPORTANT NOTE: the y data format is (x1, x2, y1, y2)
print(data[25, range(0, 281250)])
