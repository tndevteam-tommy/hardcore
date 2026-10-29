# HARDCORE — Technical Design Document & Architecture Specification

This document provides the complete mathematical, architectural, and algorithmic specification for the **Hardcore** parkour engine.

---

# 1. Architectural Pipeline & Tick Lifecycle

The engine operates on a fixed-timestep physics update with variable-timestep visual interpolation (semi-fixed timestep loop).

```
 ┌─────────────────────────────────────────────────────────────┐
 │                      Main Loop Tick                         │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 1. Event Polling & Input Buffer Update                      │
 │    • Gather Pygame keyboard & mouse events.                 │
 │    • Push inputs with timestamps into Action Buffer.        │
 │    • Expire actions older than 150ms.                       │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 2. Physics & Kinematic Character Controller (Fixed dt)      │
 │    • Accumulate frame delta time.                           │
 │    • While accumulator >= FIXED_DT (e.g. 1/60s):            │
 │      a. Update FSM state (Ground, Air, Hang, Mantle, Dyno). │
 │      b. Resolve buffered input combos against current state. │
 │      c. Apply gravity, momentum, and variable jump lift.    │
 │      d. Perform Swept-AABB collision vs static blocks.      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 3. Spatial Rail Query & Soft-Lock Reticle Solver            │
 │    • Query Ledge Graph within frustum and distance cone.    │
 │    • Compute screen-space projections of rail segments.     │
 │    • Update reticle spring-damper position & target lock.   │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 4. Camera Kinematics & View Matrix                          │
 │    • Integrate yaw and pitch; apply Dutch roll & head-bob.  │
 │    • Lerp dynamic FOV based on sprint/mantle state.         │
 │    • Construct View and Projection matrices via NumPy.      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 5. Procedural First-Person Arm, Hand, and Finger Solver     │
 │    • Evaluate base procedural locomotion sway / pump.       │
 │    • If targeted/hanging: Anchor wrists via 2-Bone IK.      │
 │    • Run finger-curling solvers (crimp vs palm-press).      │
 │    • Build matrix palette (Uniform array mat4 u_Bones[]).   │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 6. Modern OpenGL 3.3 Core Render Pass                       │
 │    • Bind framebuffers, clear color & depth buffers.        │
 │    • Render static procedural world (Batched VAO).          │
 │    • Render emissive ledge highlight strips.                │
 │    • Render skinned arm/hand/finger meshes (Bone Palette).  │
 │    • Render screen-space HUD & Contextual Reticle.          │
 │    • Swap window buffers (pygame.display.flip()).           │
 └─────────────────────────────────────────────────────────────┘
```

---

# 2. Input Buffering & Action-Combo System

Human reaction times require an input buffering window so fast parkour combos register reliably without requiring frame-perfect button presses.

### 2.1 Action Buffer Data Structure
* The input manager maintains a queue of `BufferedAction` structs:
  Action = { action_id, timestamp, consumed_flag }
* Buffer duration window: tau_buffer = 0.15 seconds (150 ms).
* At timestamp t_now, any action with (t_now - timestamp) > tau_buffer is purged.

### 2.2 Combo Resolution Matrix
The controller tests for combo chords hierarchically before single-key actions:

| Action Priority | Buffered Key Combo | Current State | Target State | Condition |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Highest)**| `Space + W + E` | `LEDGE_HANG` | `DYNO_UP` | Wall face clear above |
| **2** | `Space + S + E` | `LEDGE_HANG` | `WALL_EJECT` | Space behind player clear |
| **3** | `Space + A + E` | `LEDGE_HANG` | `DYNO_LEFT` | Lateral trajectory clear |
| **4** | `Space + D + E` | `LEDGE_HANG` | `DYNO_RIGHT`| Lateral trajectory clear |
| **5** | `W + E` | `LEDGE_HANG` | `MANTLE_VAULT` | Current rail is `MANTLE_TOP` |
| **6** | `W + E` | `LEDGE_HANG` | `CREVASSE_UP` | Higher crevasse in reach |
| **7** | `S + E` | `LEDGE_HANG` | `CREVASSE_DOWN`| Lower crevasse in reach |
| **8** | `S + E` | `LEDGE_HANG` | `AIRBORNE_FALL`| No lower crevasse (Drop) |
| **9** | `A + E` / `D + E`| `LEDGE_HANG` | `SHIMMY` | Target within rail bounds |
| **10** | `E` | `AIRBORNE` / `GROUND` | `LEDGE_HANG` | Valid rail in reach cone |
| **11** | `Space` (Hold) | `GROUND` | `AIRBORNE_JUMP` | t_hold <= 0.25s |
| **12 (Lowest)**| `Space` (Tap) | `GROUND` | `AIRBORNE_JUMP` | Instant initial impulse |

---

# 3. Parametric Ledge & Crevasse Graph System

Rather than using computationally expensive, jitter-prone real-time mesh raycasting, the environment builds an explicit **Parametric Ledge Graph** during level generation.

### 3.1 Parametric Rail Definition
Every interactable edge is defined by a directed 3D line segment:
L(s) = P1 + s * (P2 - P1),  s in [0, 1]

```
          P1 ─────────────────────── P2   <-- Rail Segment
             │                     │
             │     Wall Face       │
             │                     │
             ▼ Normal (N)          ▼
```

Each rail contains:
* `p1`: R^3 starting vertex.
* `p2`: R^3 ending vertex.
* `normal` (N): Normalized horizontal vector pointing outward, perpendicular to the obstacle face.
* `tangent` (T): Normalized unit vector along the edge: T = (P2 - P1) / ||P2 - P1||.
* `rail_type`:
  * `HEAD_CREVASSE`: Wall indentation at y ≈ 2.0 m.
  * `JUMP_CREVASSE`: Wall indentation at y ≈ 3.8 m.
  * `MANTLE_TOP`: Uppermost edge of an obstacle with a traversable top surface.
* `thickness`: Rail lip depth (used by procedural fingers to decide crimp curl radius).

### 3.2 Projection & Spatial Clamping
When the player hangs or shimmies, their position along the rail is parameterized by scalar s:
s = ((P_player - P1) · (P2 - P1)) / ||P2 - P1||^2
s_clamped = clamp(s, s_min, s_max)
where s_min and s_max account for the player's physical shoulder width (0.35 m) preventing hands from hanging off the void past obstacle corners.

### 3.3 Anchor Standoff Transform
The player's body and eye position are locked to the rail anchor:
P_eye = L(s_clamped) - N * d_standoff - y_hat * h_hang
where d_standoff ≈ 0.28 m (distance from wall) and h_hang ≈ 0.35 m (vertical distance between eyes and hands gripping the ledge).

---

# 4. Contextual Soft-Lock Reticle

The reticle provides continuous visual feedback of interactable edges without requiring pixel-perfect crosshair aiming.

```
       [ Screen Viewport ]
       ┌────────────────────────┐
       │                        │
       │       (  +  )          │   <-- Default Center Crosshair
       │         \              │
       │          \ (Magnetic)  │
       │           ▼            │
       │         [ < > ]        │   <-- Soft-Locked Target on Nearest Edge
       │    ════════════════    │   <-- Highlighted Ledge Rail
       │                        │
       └────────────────────────┘
```

### 4.1 Targeting Mathematics
1. **Candidate Query**: For all rails L_i within reach distance (d <= 4.5 m):
   * Sample the closest point on the segment Q_i in [P1, P2] to the player's camera position.
   * Compute vector D_i = Q_i - P_cam.
   * Compute forward alignment: cos(theta) = (D_i · V_forward) / ||D_i||.
   * Discard candidate if theta > 35° or if player is facing the wrong side of the wall (D_i · N_i < 0).
2. **Screen-Space Projection**:
   Project Q_i using the Model-View-Projection (MVP) matrix into Normalized Device Coordinates (NDC):
   v_clip = M_proj * M_view * [Q_i, 1.0]^T
   v_ndc = (1 / w_clip) * [x_clip, y_clip]^T
   v_screen = [((x_ndc + 1) / 2) * W, ((1 - y_ndc) / 2) * H]^T
3. **Magnetic Spring Smoothing**:
   The reticle position R does not jump instantaneously. It uses a critically damped spring-damper lerp:
   R_(t+dt) = R_t + (v_screen - R_t) * (1 - exp(-lambda * dt))
   When no ledge is targeted, R springs back to the screen center (W/2, H/2).

### 4.2 Visual States
* **Neutral**: Small, translucent circular dot at screen center.
* **Tracking (Out of Reach)**: Expanded bracket [ ] tracking the edge point, faint cyan color (d in [1.8 m, 4.5 m]).
* **Locked (In Reach)**: Tight diamond snap, high-intensity emissive amber glow (d <= 1.8 m). Signals that pressing `E` will instantly anchor.

---

# 5. First-Person Procedural Arm, Hand, and Finger Skeletal System

The player’s hands, fingers, and arms are drawn in the first-person view, visually demonstrating realistic grips on rails, crevasses, and mantle tops.

### 5.1 Skeletal Hierarchy & Bone Layout
The rig consists of two symmetrical arm hierarchies anchored to an inertially stabilized camera rig:

```
Camera Root (Floating Eye Base)
 └── Chest / Clavicle Base
      ├── Left Clavicle  ──> Left Shoulder  ──> Left Elbow  ──> Left Wrist
      │                                                           └── Left Hand (Metacarpals)
      │                                                                ├── Thumb  [Proximal, Distal]
      │                                                                ├── Index  [Proximal, Inter, Distal]
      │                                                                ├── Middle [Proximal, Inter, Distal]
      │                                                                ├── Ring   [Proximal, Inter, Distal]
      │                                                                └── Pinky  [Proximal, Inter, Distal]
      │
      └── Right Clavicle ──> Right Shoulder ──> Right Elbow ──> Right Wrist
                                                                  └── (Identical Finger Hierarchy)
```
* **Total Joint Count**: 2 * (1 Clavicle + 1 Shoulder + 1 Elbow + 1 Wrist + 1 Palm + 14 Finger Phalanges) = 38 joints.

### 5.2 Inertial Clavicle / Shoulder Anchor
Shoulder positions are offset from camera position P_cam:
S_left = P_cam + R_cam * [-d_shoulder_x, -d_shoulder_y, d_shoulder_z]^T + J_lag
where J_lag is an inertial lag vector that trails camera rotation velocity, providing visual weight and secondary motion during quick turns.

### 5.3 Analytical Two-Bone Inverse Kinematics (IK)
For each arm, target wrist position W is known (computed from ledge rail contact points or procedural locomotion swing).

```
   Shoulder (S)
       o
      / \
     /   \  Upper Arm (length L1)
    /     \
   o       o Elbow (E) [Constrained by Pole Vector]
           /
          / Forearm (length L2)
         /
        o Wrist (W)
```

1. **Distance Clamping**:
   D = W - S,  d = ||D||
   d_solve = min(d, 0.999 * (L1 + L2))
2. **Elbow Angle via Law of Cosines**:
   cos(alpha) = (L1^2 + d_solve^2 - L2^2) / (2 * L1 * d_solve)
   cos(beta)  = (L1^2 + L2^2 - d_solve^2) / (2 * L1 * L2)
   where alpha is the angle at the shoulder relative to D, and beta is the interior elbow flex angle.
3. **Pole Vector Plane Projection**:
   To prevent the elbow from flipping erratically, an anatomical **Pole Vector** P_pole is defined (pointing down and outward from the torso).
   * Construct normal to the limb plane: N_arm = (D x P_pole) / ||D x P_pole||
   * Construct planar perpendicular: U = N_arm x (D / d_solve)
   * Elbow position E:
     E = S + L1 * (cos(alpha) * (D / d_solve) + sin(alpha) * U)
4. **Bone Transforms**:
   * Upper arm rotation aligns along vector (E - S).
   * Forearm rotation aligns along vector (W - E).

### 5.4 Hand Orientation & Finger Contact Solvers
The wrist orientation matrix R_wrist aligns with the obstacle geometry:
* **Forward / Dorsal Vector**: Aligns with the wall normal -N_wall.
* **Lateral Vector**: Aligns with the ledge tangent T_rail.

#### Pose A: Crimp Grip (Crevasses & Thin Ledges)
When gripping a thin crevasse lip:
* **Metacarpophalangeal (MCP / Knuckle) Joint**: Flexes +45° forward over the lip.
* **Proximal Interphalangeal (PIP) Joint**: Flexes +90° downward, driving fingertips into the crevasse pocket.
* **Distal Interphalangeal (DIP) Joint**: Flexes +15° backward (hyper-extension crimp lock).
* **Thumb**: Abducts and presses flat against the vertical wall face below the crevasse, countering fingertip downward pressure.

```
          Lip Surface ═════════╗
     Fingers (Crimp):  MCP \   ║ <-- Crevasse Pocket
                           PIP │
                               DIP \
     Wall Surface ─────────────║────
                      Thumb: ( O )
```

#### Pose B: Mantle Palm-Press (Flat Rooftops & Top Edges)
When mantling or pushing downward on top of a surface:
* Wrist extends upward +30° to +45°.
* Palm flattens onto the horizontal surface (y = y_ledge).
* All five fingers splay outward radially by 10° - 15° and lie flat along the horizontal plane.

#### Pose C: Pre-Reach Anticipation (In-Flight near Rail)
When airborne and within 1.8 m of a targeted rail:
* Arms extend forward toward the projected contact points.
* Hands open wide: MCP joints flex -15° (flared open), fingers slightly curved, thumbs spread outward awaiting contact.

#### Pose D: Locomotion Sway & Sprint Pump (Ground Movement)
When not interacting with ledges:
* Hands return to natural first-person rest position.
* A procedural running cycle drives rhythmic opposing vertical and depth swings:
  y_left = A_y * sin(omega * t),      z_left = A_z * cos(omega * t)
  y_right = A_y * sin(omega * t + pi), z_right = A_z * cos(omega * t + pi)
* During sprint, frequency omega scales up by 1.8x and wrist rotations lean forward into an aggressive athletic pump.

### 5.5 GPU Vertex Skinning Pipeline
Rather than updating thousands of hand mesh vertices on the CPU in Python, bone transformations are solved as a flat palette array:
```glsl
// GLSL 3.3 Core uniform block
uniform mat4 u_Bones[38];
```
Each vertex in the arm/hand mesh stores:
* `in vec3 a_Position;`
* `in vec3 a_Normal;`
* `in vec2 a_UV;`
* `in vec4 a_BoneIDs;`    // Up to 4 bone influences
* `in vec4 a_Weights;`    // Normalized blend weights (sum w = 1.0)

Linear Blend Skinning (LBS) runs entirely on the GPU in `arm.vert`:
P_skinned = sum_{k=0}^3 a_Weights[k] * (u_Bones[a_BoneIDs[k]] * P_local)

---

# 6. Kinematic Character Controller & Momentum Physics

```
               ┌───────────────────────────────┐
               │    Base Locomotion Forces     │
               └───────────────┬───────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   [ Ground / Sprint ]                     [ Airborne ]
   • Acceleration: 32.0 m/s²               • Air Control: 6.0 m/s²
   • Dynamic Friction: 12.0 m/s²           • Horizontal Drag: 0.25 m/s²
   • V_max: 5.0 m/s (Walk)                 • Gravity: -19.62 m/s²
            9.5 m/s (Sprint)               • Sustained Jump Lift: +14.0 m/s²
                                             (Window: t <= 0.25s)
```

### 6.1 Velocity Integration
Player velocity V = [v_x, v_y, v_z]^T is integrated via Symplectic Euler:
V_(t+dt) = V_t + a_net * dt
P_(t+dt) = P_t + V_(t+dt) * dt

### 6.2 Variable Jump System
When `Spacebar` is pressed on the ground:
1. Ground flag cleared; initial instantaneous vertical impulse applied:
   v_y = 5.2 m/s
2. Jump timer initialized: t_jump = 0.25 s.
3. During successive frames:
   * If `Spacebar` is held and t_jump > 0:
     v_y = v_y + (g_lift * dt),  g_lift = +14.0 m/s²
     t_jump = t_jump - dt
   * If `Spacebar` is released:
     t_jump = 0  (Instantly cuts upward lift)
4. Full downward gravity (g = -19.62 m/s²) applies at all times, producing a snappy, non-floaty parabolic trajectory.

### 6.3 Swept-AABB Collision Resolution
The player collision body is an Axis-Aligned Bounding Box:
* Dimensions: Width = 0.6 m, Depth = 0.6 m, Height = 1.8 m.
* For each movement step delta_P = V * dt:
  1. Broadphase: Expand player AABB by delta_P and query overlapping obstacle boxes.
  2. Narrowphase: Perform 1D Swept Minkowski tests along each axis sequentially (X -> Z -> Y).
  3. On contact with surface normal n_hat:
     * Slide resolution: Zero out velocity component along normal:
       V = V - (V · n_hat) * n_hat
     * If n_hat == [0, 1, 0]^T: Grounded state flagged true.

---

# 7. Parkour Action Trajectories & Splines

### 7.1 Mantle Vault Trajectory
When a player initiates a mantle (`W + E` on `MANTLE_TOP` rail), the state machine takes direct kinematic control and drives the player root along a 2-stage cubic Hermite curve:

P(u) = (2u^3 - 3u^2 + 1)*P0 + (u^3 - 2u^2 + u)*m0 + (-2u^3 + 3u^2)*P1 + (u^3 - u^2)*m1,  u in [0, 1]

* **Stage 1: Vertical Hoist (t in [0, 0.28s])**:
  * P0 = P_hang
  * P1 = L(s) + y_hat * (h_waist + 0.1m) - N_wall * 0.1m
  * Arms transition from crimp to palm-press.
* **Stage 2: Forward Step-Through (t in [0.28s, 0.45s])**:
  * Player steps onto the top flat surface:
    P_final = L(s) - N_wall * 0.8m + y_hat * h_stand
  * State transitions seamlessly into `GROUND` locomotion, retaining forward velocity (v_exit = 4.0 m/s).

### 7.2 Dyno Jump & Wall Ejection Trajectories
When pushing off walls, velocities are set instantaneously and a **grab lockout timer** (tau_lockout = 0.22s) prevents re-grabbing the same rail immediately:
* **Dyno Up (`Space + W + E`)**:
  V = y_hat * 8.5 m/s - N_wall * 1.2 m/s
  *Both arms push downward explosively against the wall; camera pitches slightly upward.*
* **Wall Eject (`Space + S + E`)**:
  V = N_wall * 7.5 m/s + y_hat * 4.5 m/s
  *Arms push straight out into open air; camera undergoes a 180° yaw blend prompt.*
* **Lateral Dyno Leap (`Space + A/D + E`)**:
  V = ±T_rail * 6.0 m/s + y_hat * 5.0 m/s + N_wall * 2.0 m/s
  *Camera rolls ±4° toward the leap direction.*

---

# 8. Modern OpenGL 3.3 Core Rendering Pipeline

The rendering system strictly enforces Modern Core Profile architecture. No legacy fixed-function calls (`glBegin`, `glVertex`, `glMatrixMode`) are used.

```
       ┌────────────────────────────────────────────────────────┐
       │                   Render Pipeline                      │
       └───────────────────────────┬────────────────────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
[ World Pass ]             [ Ledge Emissive Pass ]      [ First-Person Arms ]
• Default Shader           • Emissive Edge Shader       • Skinned Bone Shader
• Batched Static Geometry  • Glowing Line Strips        • Uniform mat4 u_Bones[38]
• Ambient + Directional    • Depth-offset to prevent    • Linear Blend Skinning
• Procedural Grid Lines      Z-fighting                 • Phong specular highlight
      │                            │                            │
      └────────────────────────────┼────────────────────────────┘
                                   │
                                   ▼
                        [ Screen-Space HUD Pass ]
                        • Orthographic 2D Projection
                        • Soft-lock magnetic reticle
                        • Emissive targeting brackets
```

### 8.1 Shader Program Specifications

#### 1. World Geometry Shader (`world.vert` / `world.frag`)
* **Inputs**: `vec3 in_Position`, `vec3 in_Normal`, `vec2 in_UV`.
* **Uniforms**: `mat4 u_Model`, `mat4 u_View`, `mat4 u_Proj`, `vec3 u_LightDir`, `vec3 u_CamPos`.
* **Fragment Logic**: Calculates directional diffuse lighting, ambient fill, and a procedural world-space anti-aliased grid pattern that highlights architectural elevation changes.

#### 2. Ledge Rail Emissive Shader (`ledge.vert` / `ledge.frag`)
* **Inputs**: `vec3 in_Position`.
* **Uniforms**: `mat4 u_MVP`, `vec3 u_GlowColor`, `float u_Alpha`.
* **Rendering**: Interactive rails are drawn as extruded thin volume strips slightly biased along surface normals (epsilon = +0.005 m) using `glPolygonOffset(-1.0, -1.0)` to eliminate Z-fighting with wall surfaces.
* **Colors**: Cyan (`#00E5FF`) for active/reachable rails; Amber (`#FF9100`) for locked/grasped rails.

#### 3. First-Person Arm Skinning Shader (`arm.vert` / `arm.frag`)
* **Inputs**: `vec3 in_Position`, `vec3 in_Normal`, `vec2 in_UV`, `vec4 in_BoneIDs`, `vec4 in_Weights`.
* **Uniforms**: `mat4 u_Bones[38]`, `mat4 u_View`, `mat4 u_Proj`, `mat4 u_Model`.

#### 4. UI / Reticle Shader (`ui.vert` / `ui.frag`)
* Rendered in Normalized Device Coordinates (NDC) with depth testing disabled (`glDisable(GL_DEPTH_TEST)`).
* Draws soft-lock brackets, range indicator rings, and center target dot.

---

# 9. Procedural World Generation Specification

The environment is generated procedurally via modular geometric blocks that automatically build the parametric ledge graph.

### 9.1 Obstacle Primitives
Levels are constructed from structured volumetric primitives:
```python
class ObstaclePrimitive:
    def __init__(self, x, y, z, width, height, depth, has_crevasses=True):
        self.bounds = AABB(min=(x, y, z), max=(x + width, y + height, z + depth))
        self.rails = []
        # 1. Top Mantle Edges (All 4 perimeter edges of top surface)
        # 2. Head Crevasses (y = 2.0m on exposed vertical faces)
        # 3. Jump Crevasses (y = 3.8m on exposed vertical faces)
```

### 9.2 Automatic Ledge Rail Extraction Algorithm
For every vertical face of an obstacle:
1. Determine face normal N (e.g. [0, 0, 1] for South face).
2. Compute horizontal tangent T = N x [0, 1, 0].
3. **Mantle Rail**: Instantiate rail along the top edge:
   P1 = C_top_left,  P2 = C_top_right,  type = MANTLE_TOP
4. **Crevasse Rails**: If wall height H >= 2.5 m:
   * Instantiate horizontal rail at y = 2.0 m, recessed into wall face by 0.08 m to represent a physical hand slot (`HEAD_CREVASSE`).
5. **Jump Rails**: If wall height H >= 4.5 m:
   * Instantiate horizontal rail at y = 3.8 m, recessed by 0.08 m (`JUMP_CREVASSE`).

### 9.3 Standard Playground Level Layout
The default procedural testing level instantiates a complete parkour gauntlet:
* **Zone 1: Locomotion & Variable Jump Runway**:
  * Low hurdles (0.8 m to 1.2 m) testing tap-hops vs full-hold jumps.
* **Zone 2: Crevasse Traverse Wall**:
  * Continuous 20 m long wall with a head-height crevasse (2.0 m) requiring grab and shimmying (`A/D + E`).
* **Zone 3: Vertical Double-Crevasse Tower**:
  * Pillar featuring a 2.0 m crevasse, a 3.8 m crevasse, and a 5.2 m mantle top, requiring chained vertical climbs (`W + E`) and dyno jumps (`Space + W + E`).
* **Zone 4: The Dyno Chasm**:
  * Two opposing towers separated by a 4.5 m gap, testing the backward Wall Eject (`Space + S + E`) across the void.
* **Zone 5: Pillar Hop Array**:
  * Four staggered square columns (1.5 m x 1.5 m x 6.0 m) spaced 3.0 m apart, testing lateral dyno leaps (`Space + A/D + E`).

---

# 10. Class Contracts & Module Breakdown

### 10.1 `math3d/`
* `Transform`:
  * `perspective(fov_deg, aspect, near, far) -> np.ndarray` (Returns 4x4 projection matrix).
  * `look_at(eye, target, up) -> np.ndarray` (Returns 4x4 view matrix).
  * `translate(v), rotate_x(a), rotate_y(a), rotate_z(a), scale(s) -> np.ndarray`.
* `Quaternion`:
  * `from_axis_angle(axis, angle_rad) -> np.ndarray` (q = [w, x, y, z]).
  * `slerp(q1, q2, t) -> np.ndarray`.
  * `to_rotation_matrix(q) -> np.ndarray` (4x4 rotation matrix).
* `Geometry`:
  * `point_segment_distance(pt, seg_a, seg_b) -> (float, float)` (Returns distance and projection factor s).
  * `ray_aabb_intersection(ray_origin, ray_dir, box_min, box_max) -> float | None`.

### 10.2 `parkour/`
* `LedgeRail`:
  * `p1: np.ndarray`, `p2: np.ndarray`, `normal: np.ndarray`, `tangent: np.ndarray`.
  * `rail_type: int`, `length: float`.
  * `project(point) -> (np.ndarray, float)`: Projects point onto segment, returns 3D anchor and clamped s.
* `LedgeSystem`:
  * `find_best_rail(camera_pos, look_dir, reach_radius) -> (LedgeRail, np.ndarray, float)`.
  * `get_rails_in_frustum(view_proj_matrix) -> list[LedgeRail]`.
* `ParkourStateMachine`:
  * States: `GROUND`, `SPRINT`, `AIRBORNE_JUMP`, `AIRBORNE_FALL`, `LEDGE_HANG`, `LEDGE_SHIMMY`, `MANTLE_VAULT`, `DYNO_LAUNCH`, `WALL_EJECT`.
  * `update(dt, input_buffer, kcc, ledge_system) -> None`.

### 10.3 `animation/`
* `TwoBoneIK`:
  * `solve(shoulder_pos, target_wrist_pos, l1, l2, pole_vector) -> (elbow_pos, q_upper, q_forearm)`.
* `FingerSolver`:
  * `solve_hand(wrist_transform, surface_normal, grip_mode) -> list[np.ndarray]` (Returns 15 relative finger joint rotation matrices).
* `FirstPersonArmRig`:
  * `update(dt, player_state, camera_transform, target_rail, target_anchor) -> None`.
  * `get_bone_palette() -> np.ndarray` (Returns 1D flattened array of 38 mat4 transforms for shader uniform upload).

### 10.4 `render/`
* `GLShader`:
  * `compile(vert_src, frag_src) -> None`.
  * `set_mat4(name, np_matrix)`, `set_vec3(name, np_vec)`, `set_float(name, val)`.
* `MeshBuffer`:
  * Manages `glGenVertexArrays`, `glGenBuffers`, `glBufferData`.
  * Configures interleaved vertex attributes (Position, Normal, UV, BoneIDs, Weights).
* `Renderer`:
  * `draw_world(world, view_matrix, proj_matrix)`.
  * `draw_ledges(ledge_system, view_matrix, proj_matrix)`.
  * `draw_arms(arm_rig, view_matrix, proj_matrix)`.
  * `draw_reticle(reticle_solver, screen_width, screen_height)`.

---

# 11. Implementation Roadmap & Milestones

1. **Milestone 1 — Core GL 3.3 Context & Math Core**:
   * Modern OpenGL windowing via Pygame.
   * Complete NumPy 3D math suite (transformations, quaternions, geometry projections).
   * Shader compilation pipeline and basic textured/grid floor.
2. **Milestone 2 — Kinematic Controller & Action Buffer**:
   * Timestamped input queue (150 ms buffer window).
   * Swept-AABB collision physics vs static procedural blocks.
   * Momentum integration, sprint FOV lerping, and 0.25s variable jump reservoir.
3. **Milestone 3 — Ledge Graph & Soft-Lock Targeting Reticle**:
   * Procedural block rail extractor (`HEAD_CREVASSE`, `JUMP_CREVASSE`, `MANTLE_TOP`).
   * Emissive ledge highlight shader pass.
   * Screen-space soft-lock magnetic reticle with distance-based bracket tightening.
4. **Milestone 4 — Parkour FSM & Traversal Mechanics**:
   * Ledge anchor snapping and clamped shimmying along rail tangents.
   * Chained multi-crevasse climbing.
   * Hermite spline root motion for the two-stage Mantle Vault.
   * Dyno jumps (`Space + W + E`), Wall Ejections (`Space + S + E`), and lateral dynos (`Space + A/D + E`).
5. **Milestone 5 — Procedural 2-Bone Arm, Hand, and Finger IK**:
   * Camera-anchored clavicle/shoulder system with rotational inertia.
   * Analytical 2-bone arm solver with pole vector constraints.
   * Procedural finger curling (crimp grip vs flat mantle palm-press vs airborne pre-reach).
   * GPU Linear Blend Skinning (LBS) via 38-bone uniform array palette.
6. **Milestone 6 — Playground Course & Polish**:
   * Build complete 5-zone parkour obstacle course.
   * Integrate camera kinematics (head bobbing, Dutch roll tilts during dynos).
   * Frame-pacing verification, friction tuning, and release polish.
