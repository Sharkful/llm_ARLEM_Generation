// Deliberately invalid OpenSCAD used by tests/integration/test_openscad_compile.py
// to verify compile_scad() reports failure (not just empty output) on real syntax errors.
union() {
    cube([0.05, 0.005, 0.02);
}
