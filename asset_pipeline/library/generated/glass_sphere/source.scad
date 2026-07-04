
// Transparent Blue Glass Sphere
// A simple sphere ~20cm in diameter, centered at the origin.

// --- Key Parameters ---
diameter = 0.20;   // Overall diameter in meters
radius   = diameter / 2;

// --- Geometry ---
// Single sphere centered at origin
sphere(r = radius, $fn = 64);
