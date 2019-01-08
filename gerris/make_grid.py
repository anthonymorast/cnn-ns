import numpy as np

<<<<<<< HEAD
xmin = -2.5
xmax = 4.5
ymin = -2.5
ymax = 2.5

multiplier = 50     # determines fidelity of data 100 -> steps 0.01, 0.02, 0.03, etc.
=======
xmin = -0.5
xmax = 3.5
ymin = -0.5
ymax = 1.5

multiplier = 20     # determines fidelity of data 100 -> steps 0.01, 0.02, 0.03, etc.
>>>>>>> b38b8a80b69e6aa3fea57b7b2e9ef3d61fe0deec

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
