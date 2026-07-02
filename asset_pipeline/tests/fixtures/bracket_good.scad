// Simple L-bracket fixture used by tests/integration/test_openscad_compile.py
// to verify the real OpenSCAD binary compiles and produces sane STL bounds.
width = 0.05;
height = 0.05;
depth = 0.02;
thickness = 0.005;

union() {
    cube([width, thickness, depth]);
    cube([thickness, height, depth]);
}
