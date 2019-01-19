import pandas as pd
import os
import sys

def getDf(filename):
    df = pd.read_csv(filename, usecols=['t', 'x', 'y', 'U', 'V'])
    
    starttime = 0
    endtime = 5
    incr = 0.02
    
    ## 56,250 columns per row since 56,250 nodes were queried when creating datasets
    maxnode = 56250
    cols = []
    for node in range(0, maxnode):
        for name in list(df):
            cols.append(str(node)+name)
        
    t = starttime
    all_rows = []
    while t <= endtime:
        t = round(t, 2)
        
        rows = df[df.t == t]
        row_data = []
        for _, row in rows.iterrows():
            row_data.append(row.t)
            row_data.append(row.x)
            row_data.append(row.y)
            row_data.append(row.U)
            row_data.append(row.V)

        all_rows.append(row_data)
        t += incr
        
    return pd.DataFrame(all_rows, columns=cols)

def printcsv(df, filename):
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    d = os.path.join('.')
    idmin = 19
    idmax = 20
    
    for i in range(idmin, idmax+1):
        if i == 12:
            continue;
        filename = 'id'+str(i)+'_data'
        print("Processing file: ", filename)
        thisDf = getDf(filename)
        printcsv(thisDf, filename+'.csv')
