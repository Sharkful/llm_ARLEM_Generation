
// ============================================================
//  Apollo 11 Command/Service Module "Columbia" — Scale Model
//  Units: meters  |  +Y up, +Z forward
//  Central axis: Y-axis  |  Origin: center of combined CSM stack
// ============================================================

// --- Key Parameters ---
// Service Module (SM)
sm_radius = 0.065;    // SM outer radius  (~3.9 m real → ~1:60 scale)
sm_height = 0.1208;   // SM height — halved from original 0.2417

// Command Module (CM)
cm_base_radius = 0.065; // CM base matches SM top radius
cm_apex_radius = 0.01; // CM nose tip radius (small, rounded cap)
cm_height = 0.0683;// CM height        (~3.5 m real → 1:60)

// SPS engine bell (Service Propulsion System)
sps_r_top = 0.014;     // bell throat radius (top, where it meets SM)
sps_r_bot = 0.028;     // bell exit radius   (wider at bottom)
sps_height = 0.04;    // bell length

// RCS quad pods (4 clusters, 90° apart, mounted on SM side)
rcs_pod_w = 0.01;     // width  (circumferential)
rcs_pod_h = 0.008;     // height (axial, along Y)
rcs_pod_d = 0.006;     // depth  (radial, sticks out from SM surface)
rcs_nozzle_r = 0.0018;  // small nozzle exit radius
rcs_nozzle_h = 0.004;   // nozzle protrusion length

// Interface ring between SM and CM
ring_thickness = 0.003;
ring_height = 0.005;

// Panel rib detail on SM surface
sm_panel_depth = 0.0015;

// ---- Derived positions (stack runs along Y axis) ----
// Stack total: SM occupies [-sm_height/2, +sm_height/2] in Y
// CM sits on top of SM, SPS hangs below SM
sm_bot_y = -sm_height / 2;
sm_top_y =  sm_height / 2;
cm_bot_y =  sm_top_y;          // CM base flush with SM top
cm_top_y =  cm_bot_y + cm_height;
sps_bot_y = sm_bot_y - sps_height; // bell bottom

// ============================================================
//  SERVICE MODULE — hollow cylinder with panel ribs
// ============================================================
module service_module() {
    color([0.80, 0.82, 0.85]) {   // silver-white
        difference() {
            // Outer skin
            translate([0, sm_bot_y, 0])
            rotate([-90, 0, 0])
                cylinder(r = sm_radius, h = sm_height,
                         center = false, $fn = 72);

            // Hollow interior cutaway (optional visual detail)
            translate([0, sm_bot_y + 0.004, 0])
            rotate([-90, 0, 0])
                cylinder(r = sm_radius - 0.004, h = sm_height - 0.008,
                         center = false, $fn = 72);
        }

        // Longitudinal panel ribs — 6 bays
        for (i = [0 : 5]) {
            rotate([0, i * 60, 0])
            translate([sm_radius, 0, 0])
                cube([sm_panel_depth * 2, sm_height * 0.80, 0.002],
                     center = true);
        }
    }
}

// ============================================================
//  INTERFACE RING — thin collar between SM top and CM base
// ============================================================
module interface_ring() {
    color([0.55, 0.55, 0.58]) {
        translate([0, sm_top_y, 0])
        rotate([-90, 0, 0])
            difference() {
                cylinder(r = sm_radius + 0.002, h = ring_height,
                         center = false, $fn = 72);
                cylinder(r = sm_radius - ring_thickness, h = ring_height,
                         center = false, $fn = 72);
            }
    }
}

// ============================================================
//  COMMAND MODULE — truncated cone + rounded apex
// ============================================================
module command_module() {
    color([0.88, 0.88, 0.88]) {   // light gray
        // Main frustum: wide base at cm_bot_y, narrow apex at cm_top_y
        translate([0, cm_bot_y, 0])
        rotate([-90, 0, 0])
            cylinder(r1 = cm_base_radius, r2 = cm_apex_radius,
                     h = cm_height, center = false, $fn = 72);

        // Rounded apex cap — sits exactly at cm_top_y
        translate([0, cm_top_y, 0])
            sphere(r = cm_apex_radius, $fn = 32);

        // Slight heat-shield base ring
        translate([0, cm_bot_y - 0.002, 0])
        rotate([-90, 0, 0])
            cylinder(r = cm_base_radius, h = 0.003,
                     center = false, $fn = 72);
    }
}

// ============================================================
//  SPS ENGINE BELL — frustum, narrow end at SM bottom
// ============================================================
module sps_engine() {
    color([0.28, 0.28, 0.28]) {   // dark gray / engine bell
        // Bell: r1 (top, at SM bottom) = sps_r_top, r2 (exit) = sps_r_bot
        // rotate([90,0,0]) points the cylinder's +h direction toward -Y
        translate([0, sm_bot_y, 0])
        rotate([90, 0, 0])
            cylinder(r1 = sps_r_top, r2 = sps_r_bot,
                     h = sps_height, center = false, $fn = 48);
    }
}

// ============================================================
//  RCS QUAD POD — one rectangular block with 4 tiny nozzles
// ============================================================
module rcs_quad_pod() {
    color([0.60, 0.62, 0.65]) {
        // Pod body, centered at the mount point
        cube([rcs_pod_w, rcs_pod_h, rcs_pod_d], center = true);

        // Nozzle pointing radially outward (+X after pod placement)
        translate([rcs_pod_w * 0.5, rcs_pod_h * 0.25, 0])
        rotate([0, 90, 0])
            cylinder(r1 = rcs_nozzle_r * 1.3, r2 = rcs_nozzle_r * 0.6,
                     h = rcs_nozzle_h, center = false, $fn = 10);

        translate([rcs_pod_w * 0.5, -rcs_pod_h * 0.25, 0])
        rotate([0, 90, 0])
            cylinder(r1 = rcs_nozzle_r * 1.3, r2 = rcs_nozzle_r * 0.6,
                     h = rcs_nozzle_h, center = false, $fn = 10);

        // Nozzle pointing axially +Y
        translate([0, rcs_pod_h * 0.5, 0])
        rotate([-90, 0, 0])
            cylinder(r1 = rcs_nozzle_r * 1.3, r2 = rcs_nozzle_r * 0.6,
                     h = rcs_nozzle_h, center = false, $fn = 10);

        // Nozzle pointing axially -Y
        translate([0, -rcs_pod_h * 0.5, 0])
        rotate([90, 0, 0])
            cylinder(r1 = rcs_nozzle_r * 1.3, r2 = rcs_nozzle_r * 0.6,
                     h = rcs_nozzle_h, center = false, $fn = 10);
    }
}

// ============================================================
//  RCS CLUSTERS — 4 quads at 90° spacing near SM top
// ============================================================
module rcs_clusters() {
    // Place quads near top of SM, spaced 90° apart at 45° offsets
    // Clamped so they stay within the shorter SM body
    rcs_y = sm_top_y - 0.015;  // ~15 mm down from SM top (was 30, halved with SM)
    for (i = [0 : 3]) {
        rotate([0, i * 90 + 45, 0])
        translate([sm_radius + rcs_pod_d * 0.5, rcs_y, 0])
            rcs_quad_pod();
    }
}

// ============================================================
//  ASSEMBLY
// ============================================================
union() {
    service_module();
    interface_ring();
    command_module();
    sps_engine();
    rcs_clusters();
}
