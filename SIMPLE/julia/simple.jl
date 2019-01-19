xmin = 0.0;
xmax = 2.0;
ymin = 0.0;
ymax = 2.0;

eps = 0.001;
nx = 41;
ny = 41;
nit = 50;
c = 1;
dx = xmax/(nx-1);
dy = ymax/(ny-1);

x = range(xmin, stop=xmax, length=nx) |> collect # deprecated linspace hack
y = range(ymin, stop=ymax, length=ny) |> collect

# physical properties
rho = 1.0     # fluid density
nu = 0.1      # fluid viscosity
F = 1         # velocity in x direction
dt = .01      # time step

#initial conditions
u = zeros(ny, nx)       # x-vel
un = zeros(ny, nx)

v = zeros(ny, nx)       # y-vel
vn = zeros(ny, nx)

p = ones(ny, nx)        # pressure
pn = ones(ny, nx)

b = zeros(ny, nx)       # boundary

while udiff > eps:
    
