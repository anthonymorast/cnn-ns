import random

nlocs = 9
xmax = 0
xmin = -2.5
ymin = -2.5
ymax = 2.5

count = 0
while count < nlocs:
    x1 = random.uniform(xmin, xmax)
    x2 = random.uniform(xmin, xmax)
    while abs(x1-x2) < 0.1:
        x2 = random.uniform(xmin, xmax)

    y1 = random.uniform(ymin, ymax)
    y2 = random.uniform(ymin, ymax)
    while abs(y1-y2) < 0.1:
        y2 = random.uniform(ymin, ymax)

    print("(X1, Y1) = (%3f, %3f) -- (X2, Y2) = (%3f, %3f)" % (x1, y1, x2, y2))
    count += 1
