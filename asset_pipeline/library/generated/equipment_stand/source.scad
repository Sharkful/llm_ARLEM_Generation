
// ── Four-Legged Rectangular Equipment Stand ──────────────────────────────────
// Units: meters  |  +Y up, +Z forward  |  base sits at Z=0 (surface object)

// ── Key parameters ────────────────────────────────────────────────────────────
stand_height      = 0.40;   // total height of the stand (m)
top_width         = 0.30;   // width  of the top platform (X)
top_depth         = 0.20;   // depth  of the top platform (Z)
top_thickness     = 0.012;  // thickness of the top shelf

leg_width         = 0.018;  // square cross-section side of each leg
foot_height       = 0.010;  // small foot pad under each leg
brace_height      = 0.10;   // height of horizontal cross-brace above ground
brace_thickness   = 0.010;  // thickness of the brace bar

// ── Derived values ────────────────────────────────────────────────────────────
leg_height  = stand_height - top_thickness - foot_height;
// Leg centres inset half a leg-width from the platform corners
lx = top_width  / 2 - leg_width / 2;
lz = top_depth  / 2 - leg_width / 2;

// ── Helper: a single leg + foot pad ──────────────────────────────────────────
module leg_with_foot(x, z) {
    translate([x, 0, z]) {
        // foot pad
        translate([-leg_width/2, 0, -leg_width/2])
            cube([leg_width, foot_height, leg_width]);
        // leg column
        translate([-leg_width/2, foot_height, -leg_width/2])
            cube([leg_width, leg_height, leg_width]);
    }
}

// ── Helper: horizontal brace (runs in X between two legs) ────────────────────
module brace_x(z_pos) {
    // spans between the two legs at z_pos
    brace_len = top_width - leg_width;
    translate([-brace_len/2, brace_height, z_pos - brace_thickness/2])
        cube([brace_len, brace_thickness, brace_thickness]);
}

// ── Helper: horizontal brace (runs in Z between two legs) ────────────────────
module brace_z(x_pos) {
    brace_len = top_depth - leg_width;
    translate([x_pos - brace_thickness/2, brace_height, -brace_len/2])
        cube([brace_thickness, brace_thickness, brace_len]);
}

// ── Assembly ──────────────────────────────────────────────────────────────────
union() {
    // Four legs (one at each corner)
    leg_with_foot( lx,  lz);
    leg_with_foot( lx, -lz);
    leg_with_foot(-lx,  lz);
    leg_with_foot(-lx, -lz);

    // Horizontal braces – two along X, two along Z
    brace_x( lz);
    brace_x(-lz);
    brace_z( lx);
    brace_z(-lx);

    // Top platform shelf
    translate([-top_width/2, stand_height - top_thickness, -top_depth/2])
        cube([top_width, top_thickness, top_depth]);
}
