
// === L-Bracket (Weathered Metal) ===
// All units in meters. Centered on origin.

// --- Key Parameters ---
width        = 0.08;   // overall width of the bracket (X)
arm_length   = 0.075;  // length of each arm (Y / Z)
thickness    = 0.008;  // material thickness
depth        = 0.08;   // depth of bracket into scene (Z / X cross-section)
hole_radius  = 0.004;  // mounting hole radius
hole_inset   = 0.015;  // hole centre distance from each end

// --- Derived ---
// Vertical arm: stands along +Y from origin
// Horizontal arm: extends along +Z from origin
// The bracket corner sits at origin; object is centred by offsetting by depth/2 in X

module l_bracket() {
    translate([-depth/2, -arm_length/2, -thickness/2])
    union() {
        // --- Vertical arm (rises in Y) ---
        difference() {
            cube([depth, arm_length, thickness]);
            // Two mounting holes on vertical arm
            translate([depth/2, hole_inset, -0.001])
                cylinder(h = thickness + 0.002, r = hole_radius, $fn = 24);
            translate([depth/2, arm_length - hole_inset, -0.001])
                cylinder(h = thickness + 0.002, r = hole_radius, $fn = 24);
        }

        // --- Horizontal arm (extends in Z, shares corner with vertical arm) ---
        translate([0, 0, thickness])
        difference() {
            cube([depth, thickness, arm_length]);
            // Two mounting holes on horizontal arm
            translate([depth/2, -0.001, hole_inset])
                rotate([-90, 0, 0])
                    cylinder(h = thickness + 0.002, r = hole_radius, $fn = 24);
            translate([depth/2, -0.001, arm_length - hole_inset])
                rotate([-90, 0, 0])
                    cylinder(h = thickness + 0.002, r = hole_radius, $fn = 24);
        }

        // --- Gusset / fillet block at the inner corner for rigidity ---
        translate([depth * 0.2, 0, thickness])
            rotate([0, 0, 0])
            linear_extrude(height = depth * 0.6)
            polygon(points = [
                [0, 0],
                [thickness * 1.5, 0],
                [0, thickness * 1.5]
            ]);

        // --- Weathering detail: surface bolt heads (raised nubs) ---
        for (bx = [depth * 0.2, depth * 0.8]) {
            // bolt on vertical face
            translate([bx, arm_length * 0.5, thickness])
                cylinder(h = 0.001, r = 0.002, $fn = 6);
            // bolt on horizontal face
            translate([bx, thickness, thickness + arm_length * 0.5])
                rotate([-90, 0, 0])
                    cylinder(h = 0.001, r = 0.002, $fn = 6);
        }
    }
}

l_bracket();
