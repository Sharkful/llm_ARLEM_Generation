
// ============================================================
//  Frosty the Snowman – parametric OpenSCAD model
//  Units: meters   +Y up   +Z forward
//  Bottom of base ball sits at Z=0 (stands on surface)
// ============================================================

// --- Key parameters ---
base_r        = 0.18;   // radius of bottom snowball
mid_r         = 0.13;   // radius of middle snowball
head_r        = 0.09;   // radius of head snowball
lump_scale    = 0.04;   // amplitude of hand-packed lumps
hat_brim_r    = 0.115;  // radius of hat brim
hat_brim_h    = 0.018;  // thickness of hat brim
hat_crown_r   = 0.075;  // radius of hat crown cylinder
hat_crown_h   = 0.17;   // height of hat crown
arm_len       = 0.22;   // half-length of each stick arm
arm_r         = 0.012;  // radius of stick arm
eye_r         = 0.018;  // radius of coal eye sphere
nose_len      = 0.07;   // length of carrot nose
nose_base_r   = 0.014;  // base radius of carrot nose
mouth_r       = 0.013;  // radius of coal mouth dots
coal_depth    = 0.005;  // how far coal sits inside head

// --- Derived vertical positions (Y axis is up) ---
base_cy   = base_r;                        // centre of base ball
mid_cy    = base_r*2 - 0.04 + mid_r;      // centre of mid ball (slightly overlapping)
head_cy   = mid_cy + mid_r + head_r - 0.03; // centre of head (slightly overlapping)
hat_y     = head_cy + head_r - 0.01;      // bottom of hat

// ============================================================
//  Helper: lumpy sphere – base sphere with small bumps
// ============================================================
module lumpy_sphere(r, lumps=8, amp=0.04) {
    // Approximate lumps by scaling slightly differently on each axis
    // and adding small spheres at random-ish positions
    union() {
        scale([1.0, 0.97, 1.02]) sphere(r=r, $fn=40);
        for (i = [0 : lumps-1]) {
            angle_a = i * (360 / lumps) + i * 7;
            angle_b = 30 + i * 15;
            translate([
                r * 0.88 * cos(angle_a) * sin(angle_b),
                r * 0.88 * cos(angle_b) * (i % 2 == 0 ? 1 : -1) * 0.3,
                r * 0.88 * sin(angle_a) * sin(angle_b)
            ])
            sphere(r = r * amp * (0.6 + 0.4 * (i % 3)), $fn=14);
        }
    }
}

// ============================================================
//  BODY – three stacked lumpy snowballs  (white)
// ============================================================
color([0.97, 0.97, 1.0]) {
    // Base ball
    translate([0, base_cy, 0])
        lumpy_sphere(base_r, lumps=10, amp=lump_scale/base_r);

    // Middle ball
    translate([0, mid_cy, 0])
        lumpy_sphere(mid_r, lumps=8, amp=lump_scale/mid_r);

    // Head ball
    translate([0, head_cy, 0])
        lumpy_sphere(head_r, lumps=6, amp=lump_scale/head_r);
}

// ============================================================
//  COAL EYES  (dark, slightly forward on head)
// ============================================================
color([0.08, 0.06, 0.06]) {
    for (side = [-1, 1]) {
        translate([
            side * head_r * 0.42,
            head_cy + head_r * 0.28,
            head_r * 0.82 - coal_depth
        ])
        sphere(r=eye_r, $fn=16);
    }
}

// ============================================================
//  COAL MOUTH – arc of five dots
// ============================================================
color([0.08, 0.06, 0.06]) {
    for (i = [-2, -1, 0, 1, 2]) {
        mouth_angle = i * 18;
        translate([
            head_r * 0.72 * sin(mouth_angle),
            head_cy - head_r * 0.18 + head_r * 0.12 * cos(mouth_angle * 1.5),
            head_r * 0.82 - coal_depth
        ])
        sphere(r=mouth_r, $fn=14);
    }
}

// ============================================================
//  CARROT NOSE  – orange cone pointing +Z
// ============================================================
color([1.0, 0.45, 0.05]) {
    translate([0, head_cy + head_r * 0.04, head_r * 0.88 - coal_depth])
    rotate([-90, 0, 0])   // tip points forward (+Z)
    rotate([90, 0, 0])
    cylinder(h=nose_len, r1=nose_base_r, r2=0.001, $fn=18);
}

// ============================================================
//  STICK ARMS – thin cylinders angled slightly upward
// ============================================================
color([0.35, 0.22, 0.10]) {
    for (side = [-1, 1]) {
        translate([side * mid_r * 0.88, mid_cy + mid_r * 0.15, 0])
        rotate([0, 0, side * 25])   // angle upward
        rotate([90, 0, 0])          // lay along X axis
        rotate([0, 90, 0])
        cylinder(h=arm_len, r=arm_r, center=true, $fn=10);
    }
}

// ============================================================
//  TOP HAT  – brim + crown  (black)
// ============================================================
color([0.07, 0.05, 0.07]) {
    // Brim
    translate([0, hat_y, 0])
    rotate([-90, 0, 0])
    cylinder(h=hat_brim_h, r=hat_brim_r, center=false, $fn=40);

    // Crown
    translate([0, hat_y + hat_brim_h, 0])
    rotate([-90, 0, 0])
    cylinder(h=hat_crown_h, r=hat_crown_r, center=false, $fn=40);
}
