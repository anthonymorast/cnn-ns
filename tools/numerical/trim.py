import pandas as pd

def trim_data(filename):
    d = pd.read_csv(filename)
    d = d.rename(columns=lambda x: x.strip())
    d.columns = [c.replace('-', '_') for c in d.columns]

    # trim the dataset based on (x, y) coords taken from ANSYS SpaceClaim
    d = d.query('y_coordinate > -4 and y_coordinate < 4 and \
                 x_coordinate > -12.5 and x_coordinate < 10')
    # drop useless columns
    d = d.drop(['rel_velocity_magnitude', 'relative_x_velocity', 'relative_y_velocity',
                'x_coordinate.1', 'y_coordinate.1', 'dx_velocity_dx', 'dy_velocity_dx',
                'dx_velocity_dy', 'dy_velocity_dy'], axis=1)

    if d.shape[0] != 40747:
        print("Incorrect number of rows...")
        exit()

    return d
