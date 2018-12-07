import sys
import os
from csv import writer
from io import BytesIO, StringIO

from trim import *

if __name__ == '__main__':
    indir = os.path.join('../..', 'ansys')
    outdir = os.path.join('.', 'data')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    done = ['re1', 're100', 're1000', 're1200', 're1400', 're2', 're250', 're2900', 're4', 're40', 're400']
    col_vals = ['x', 'y', 'vel_mag', 'x_vel', 'y_vel', 'time']
    df = None

    # use this to store CSV in memory (much faster than appending to the df)
    output = StringIO()
    csv_writer = writer(output)
    flat_cols = None

    for filename in os.listdir(indir):
        if os.path.isdir(os.path.join(indir,filename)) and filename.startswith('re') and filename not in done:
            path = os.path.join(indir, filename, 'data')
            re = int(filename[2:])
            for f in os.listdir(path):
                fname = os.path.join(path, f)
                time = int(fname[-4:])
                print("Processing file:", fname)
                d = trim_data(fname)

                cols = []
                data = []
                for index, row in d.iterrows():
                    node = int(row['nodenumber'])
                    cols.append([(c+str(node)) for c in col_vals])
                    row_data = [row['x_coordinate'], row['y_coordinate'],
                                 row['velocity_magnitude'], row['x_velocity'],
                                 row['y_velocity'], time]
                    data.append(row_data)
                # flatten columns and row data
                if flat_cols is None:
                    flat_cols = [item for sublist in cols for item in sublist]
                    flat_cols.append('re')
                    # csv_writer.writerow(flat_cols)

                flat_data = [item for sublist in data for item in sublist]
                flat_data.append(re)
                csv_writer.writerow(flat_data)
            output.seek(0)
            df = pd.read_csv(output, low_memory=False)
            df.columns = flat_cols
            path = os.path.join(outdir, filename)
            if not os.path.exists(path):
                os.makedirs(path)
            df.to_csv(os.path.join(path, 'data.csv'), index=False)
            output.seek(0)
            output.truncate(0)

    # output.seek(0)
    # df = pd.read_csv(output, engine='python')
    # df.columns = flat_cols
    # df.to_csv('data.csv', index=False)
