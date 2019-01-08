import numpy as np

xmin = 0
xmax = 4.5
ymin = -2.5
ymax = 2.5

multiplier = 50     # determines fidelity of data 100 -> steps 0.01, 0.02, 0.03, etc.

ysteps = abs(ymin-ymax)*multiplier + 1
xsteps = abs(xmin-xmax)*multiplier + 1

print(ysteps, xsteps)

y = np.linspace(ymin, ymax, ysteps)
x = np.linspace(xmin, xmax, xsteps)
z = 0

locs = open("positions", 'w')
for i in x:
    for j in y:
        locs.write("%.2f %.2f %.2f\n" % (i, j, z))
