
// ── L-Shaped Mounting Bracket ──────────────────────────────────────────────
// Units: meters.  Origin centred on the bracket's bounding-box mid-point.
// +Y = up,  +Z = forward.

// ── Key parameters ─────────────────────────────────────────────────────────
width          = 0.12;    // overall width (X)  – ~12 cm
flange_h       = 0.050;   // height of the vertical flange (Y)
base_d         = 0.050;   // depth  of the horizontal base leg (Z)
thickness      = 0.006;   // material thickness of both legs
hole_dia       = 0.005;   // screw-hole diameter
hole_inset     = 0.012;   // distance from outer edge to hole centre
fillet_r       = 0.008;   // corner fillet radius (chunkier than before)
fn             = 32;      // cylinder facets

// ── Derived ────────────────────────────────────────────────────────────────
total_y = flange_h + thickness;
total_z = base_d   + thickness;

// Shift so bounding box is centred on origin
ox = -width / 2;
oy = -total_y / 2;
oz = -total_z / 2;

// Inner-corner fillet: sits at the junction of the two legs on the inside
// The fillet is a quarter-round rod spanning the full width (X axis).
// It occupies the concave corner: Y = oy (bottom of flange base),
// Z = oz + total_z - thickness (back face of base leg = front face of flange).
module corner_fillet() {
    // A cylinder lying along X, radius = fillet_r, trimmed to a quarter circle
    // by intersecting with a cube that fills the inner-corner quadrant.
    translate([ox, oy + thickness, oz + total_z - thickness])
    intersection() {
        // Full cylinder along X
        rotate([0, 90, 0])
            cylinder(r = fillet_r, h = width, $fn = fn);
        // Keep only the quadrant that fills the concave corner
        // (+Y, -Z relative to the cylinder axis)
        cube([width, fillet_r, fillet_r]);
    }
}

module bracket() {
    union() {
        // ── Horizontal base leg (lies in XZ plane, at the bottom) ──────────
        translate([ox, oy, oz])
            cube([width, thickness, total_z]);

        // ── Vertical flange (rises in Y from the back edge of base leg) ────
        translate([ox, oy, oz + total_z - thickness])
            cube([width, total_y, thickness]);

        // ── Inner-corner fillet ────────────────────────────────────────────
        corner_fillet();
    }
}

module screw_holes() {
    // ── Three holes on the vertical flange ─────────────────────────────────
    // Positions: left, centre, right
    for (sx = [-1, 0, 1]) {
        x_pos = (sx == 0) ? 0 : sx * (width / 2 - hole_inset);
        translate([x_pos,
                   oy + hole_inset,
                   oz + total_z + 0.001])
            cylinder(h = thickness * 2 + 0.002,
                     d = hole_dia,
                     center = true,
                     $fn = fn);
    }

    // ── Three holes on the horizontal base leg ──────────────────────────────
    for (sx = [-1, 0, 1]) {
        x_pos = (sx == 0) ? 0 : sx * (width / 2 - hole_inset);
        translate([x_pos,
                   oy - 0.001,
                   oz + hole_inset])
            rotate([90, 0, 0])
            cylinder(h = thickness * 2 + 0.002,
                     d = hole_dia,
                     center = true,
                     $fn = fn);
    }
}

difference() {
    bracket();
    screw_holes();
}
