
// Apollo 11 Command Module "Columbia" - Scale Model
// Truncated cone body, rounded heat shield base, docking probe at apex
// Units: meters | +Y up, +Z forward | centered on origin

// --- Key Parameters ---
cm_height = 0.11;   // Height of the main conical body (m)
cm_base_radius = 0.09;  // Radius at the wide (heat shield) base (m)
cm_top_radius = 0.022;  // Radius at the truncated apex end (m)

hs_thickness = 0.008;  // Heat shield dome thickness/depth (m)
hs_radius        = cm_base_radius + 0.002; // Heat shield rim radius, slightly wider

probe_radius = 0.004;  // Radius of docking probe cylinder (m)
probe_length = 0.022;  // Length of docking probe (m)
probe_tip_r = 0.003;  // Radius of probe tip sphere (m)

probe_neck_r = 0.003;  // Thin neck below tip
probe_neck_len = 0.008;

// --- Derived geometry ---
// The model stands with heat shield at bottom (Z=0), apex+probe pointing up (+Y).
// We model along the Y axis:
//   Y=0              → bottom of heat shield
//   Y=hs_thickness   → base of cone
//   Y=hs_thickness + cm_height → top of truncated cone (apex)
//   Above that        → docking probe

total_height = hs_thickness + cm_height + probe_neck_len + probe_length + probe_tip_r;

module heat_shield() {
    // Oblate dome at the base: use scale on a sphere to flatten it
    // Centered so its flat face is at Y=hs_thickness, dome bulges toward Y=0
    translate([0, hs_thickness, 0])
    rotate([-90, 0, 0])           // sphere Z→Y, dome goes down (−Y)
    scale([1, 1, 0.38])           // flatten the sphere into a shallow dome
    sphere(r = hs_radius, $fn = 64);
}

module cone_body() {
    // Truncated cone: base at Y=hs_thickness, apex at Y=hs_thickness+cm_height
    translate([0, hs_thickness, 0])
    rotate([-90, 0, 0])           // cylinder axis along +Y
    cylinder(r1 = cm_base_radius, r2 = cm_top_radius,
             h  = cm_height, $fn = 72);
}

module docking_probe() {
    apex_y = hs_thickness + cm_height;

    // Neck (tapered cylinder from cone top to probe)
    translate([0, apex_y, 0])
    rotate([-90, 0, 0])
    cylinder(r1 = cm_top_radius * 0.6, r2 = probe_radius,
             h  = probe_neck_len, $fn = 32);

    // Main probe shaft
    translate([0, apex_y + probe_neck_len, 0])
    rotate([-90, 0, 0])
    cylinder(r = probe_radius, h = probe_length, $fn = 24);

    // Probe tip sphere
    translate([0, apex_y + probe_neck_len + probe_length, 0])
    sphere(r = probe_tip_r, $fn = 24);
}

// --- Assemble ---
// Shift whole model so midpoint of total height sits at Y=0 (origin-centered)
translate([0, -total_height / 2, 0])
union() {
    heat_shield();
    cone_body();
    docking_probe();
}
