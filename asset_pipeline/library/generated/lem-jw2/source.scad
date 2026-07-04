
// ============================================================
//  Apollo 11 Command/Service Module "Columbia" — Scale Model
//  Units: meters  |  +Y up, +Z forward
//  Origin: centered on the combined CSM stack
// ============================================================

// --- Key Parameters ---
// Service Module (SM)
sm_radius = 0.065;   // SM outer radius  (~3.9 m real → 1:60 scale ≈ 0.065 m)
sm_height = 0.2417;   // SM height         (~14.5 m real → 1:60)
sm_wall = 0.004;   // SM skin thickness

// Command Module (CM)
cm_base_radius = 0.065;   // CM base matches SM radius
cm_apex_radius = 0.012;   // CM nose tip radius
cm_height = 0.0583;   // CM height         (~3.5 m real → 1:60)

// Service Propulsion System (SPS) nozzle
sps_nozzle_r_top = 0.013;
sps_nozzle_r_bot = 0.023;
sps_nozzle_h = 0.0333;   // ~2 m real

// Reaction Control System (RCS) thrust quads
rcs_pod_w = 0.008;
rcs_pod_d = 0.012;
rcs_pod_h = 0.005;
rcs_nozzle_r = 0.0018;
rcs_nozzle_h = 0.004;

// SM panel details
sm_panel_depth = 0.0015;

// Derived
sm_bottom_z = -sm_height / 2;
sm_top_z    =  sm_height / 2;
cm_bottom_z =  sm_top_z;
cm_top_z    =  cm_bottom_z + cm_height;

// ============================================================
//  HELPERS
// ============================================================
module rcs_nozzle() {
    rotate([90, 0, 0])
        cylinder(r1=rcs_nozzle_r*1.3, r2=rcs_nozzle_r*0.6,
                 h=rcs_nozzle_h, center=false, $fn=12);
}

// One RCS quad pod (4 nozzles, two axes)
module rcs_quad() {
    // Pod body
    cube([rcs_pod_w, rcs_pod_h, rcs_pod_d], center=true);
    // Four nozzles: +Y, -Y, +Z, -Z faces
    translate([0,  rcs_pod_h/2, 0]) rotate([ 90,0,0]) cylinder(r1=rcs_nozzle_r*1.2, r2=rcs_nozzle_r*0.5, h=rcs_nozzle_h, $fn=10);
    translate([0, -rcs_pod_h/2-rcs_nozzle_h, 0]) rotate([-90,0,0]) cylinder(r1=rcs_nozzle_r*1.2, r2=rcs_nozzle_r*0.5, h=rcs_nozzle_h, $fn=10);
    translate([0,  0,  rcs_pod_d/2]) cylinder(r1=rcs_nozzle_r*1.2, r2=rcs_nozzle_r*0.5, h=rcs_nozzle_h, $fn=10);
    translate([0,  0, -rcs_pod_d/2-rcs_nozzle_h]) rotate([180,0,0]) cylinder(r1=rcs_nozzle_r*1.2, r2=rcs_nozzle_r*0.5, h=rcs_nozzle_h, $fn=10);
}

// ============================================================
//  SERVICE MODULE
// ============================================================
module service_module() {
    difference() {
        // Main SM cylinder
        translate([0, sm_bottom_z, 0])
        rotate([90, 0, 0]) rotate([0, 0, 0])
        rotate([-90, 0, 0])
            cylinder(r=sm_radius, h=sm_height, center=false, $fn=64);

        // Hollow interior (weight-saving cutaway suggestion)
        translate([0, sm_bottom_z + sm_wall, 0])
        rotate([-90, 0, 0])
            cylinder(r=sm_radius - sm_wall, h=sm_height - sm_wall*2,
                     center=false, $fn=64);

        // SM panel recesses — 6 equally-spaced longitudinal bays
        for (i = [0:5]) {
            rotate([0, i*60, 0])
            translate([sm_radius - sm_panel_depth, 0, 0])
            rotate([-90, 0, 0])
                cube([sm_radius*0.55, sm_height*0.55, sm_panel_depth*3],
                     center=true);
        }
    }
}

// ============================================================
//  COMMAND MODULE
// ============================================================
module command_module() {
    // Blunt-cone frustum built as a cylinder+sphere blend
    translate([0, cm_bottom_z, 0])
    rotate([-90, 0, 0]) {
        // Main cone body
        cylinder(r1=cm_base_radius, r2=cm_apex_radius,
                 h=cm_height, center=false, $fn=64);
        // Rounded apex cap
        translate([0, 0, cm_height])
            sphere(r=cm_apex_radius, $fn=32);
        // Slightly domed base heat-shield disk
        cylinder(r=cm_base_radius, h=0.004, center=false, $fn=64);
    }
}

// ============================================================
//  SPS ENGINE NOZZLE
// ============================================================
module sps_nozzle() {
    translate([0, sm_bottom_z, 0])
    rotate([90, 0, 0])   // point -Y (downward)
        cylinder(r1=sps_nozzle_r_top, r2=sps_nozzle_r_bot,
                 h=sps_nozzle_h, center=false, $fn=48);
}

// ============================================================
//  SM FAIRINGS / PANELS (raised rectangular ribs)
// ============================================================
module sm_fairings() {
    for (i = [0:5]) {
        rotate([0, i*60, 0])
        translate([sm_radius, 0, 0])
        rotate([0, 0, 0])
            cube([sm_panel_depth*2, sm_height*0.50, 0.003], center=true);
    }
}

// ============================================================
//  RCS QUADS — 4 clusters, spaced 90° apart
//  Mounted near the +Y (top) end of the SM
// ============================================================
module rcs_clusters() {
    rcs_y = sm_top_z - 0.018;
    for (i = [0:3]) {
        rotate([0, i*90 + 45, 0])
        translate([sm_radius + rcs_pod_w*0.5, rcs_y, 0])
            rcs_quad();
    }
}

// ============================================================
//  UMBILICAL / SM–CM INTERFACE RING
// ============================================================
module interface_ring() {
    translate([0, sm_top_z, 0])
    rotate([-90, 0, 0])
        difference() {
            cylinder(r=sm_radius + 0.002, h=0.005, center=false, $fn=64);
            cylinder(r=sm_radius - sm_wall, h=0.005, center=false, $fn=64);
        }
}

// ============================================================
//  ASSEMBLY
// ============================================================
union() {
    service_module();
    command_module();
    sps_nozzle();
    sm_fairings();
    rcs_clusters();
    interface_ring();
}
