// Gmsh project created on Wed Oct 03 15:19:36 2018
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, -.5, 0, 1.0};
//+
Point(3) = {0, .5, 0, 1.0};
//+
Point(4) = {-2, -2, 0, 1.0};
//+
Point(5) = {-2, 2, 0, 1.0};
//+
Point(6) = {-4.2, -0.5, 0, 1.0};
//+
Recursive Delete {
  Point{4}; 
}
//+
Point(7) = {-3.8, 0.7, 0, 1.0};
//+
Point(8) = {-3.8, 0.7, 0, 1.0};
//+
Recursive Delete {
  Point{7}; 
}
//+
Recursive Delete {
  Point{8}; 
}
//+
Point(7) = {5, -2, 0, 1.0};
//+
Point(8) = {5, 2, 0, 1.0};
//+
Recursive Delete {
  Point{5}; 
}
//+
Recursive Delete {
  Point{6}; 
}
//+
Point(9) = {-2, -2, 0, 1.0};
//+
Point(10) = {-2, 2, 0, 1.0};
//+
Line(1) = {9, 7};
//+
Line(2) = {7, 8};
//+
Line(3) = {8, 10};
//+
Line(4) = {10, 9};
//+
Circle(5) = {2, 1, 3};
//+
Circle(6) = {3, 1, 2};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Curve Loop(2) = {6, 5};
//+
Plane Surface(1) = {1, 2};
//+
Physical Curve(1) = {4};
//+
Physical Curve(2) = {2};
//+
Physical Curve(3) = {3, 1};
//+
Physical Curve(4) = {6, 5};
//+
Physical Surface(5) = {1};
