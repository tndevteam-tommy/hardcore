# HARDCORE — 3D Parkour Engine (Pygame + Modern OpenGL 3.3 Core)

**Hardcore** is a first-person, momentum-driven parkour engine written in Python utilizing **Pygame**, **PyOpenGL (Modern 3.3 Core Profile)**, and **NumPy**. Inspired by the tactile, kinetic traversal of *Dying Light* and *Mirror's Edge*, the engine treats every surface, edge, and crevasse as an interactive climbing entity.

Hardcore renders a full first-person dynamic skeletal arm, hand, and finger rig driven by analytical Two-Bone Inverse Kinematics (IK) and procedural finger-curling solvers. Players witness their hands realistically reach, grip, crimp, shimmy, and push off procedural surfaces, guided by a contextual, soft-lock reticle.

---

## 1. Game Design & Traversal Mechanics

Traversal in *Hardcore* is continuous, tactile, and momentum-conserving. Every maneuver chains directly into the next.

### 1.1 Kinetic Movement & Sprinting
* **Locomotion**: WASD movement with distinct acceleration curves, friction dampening, and preserved air inertia.
* **Sprint (`Left Shift`)**: Accelerates player velocity from 5.0 m/s to 9.5 m/s. Sprinting expands camera Field of View (FOV) dynamically from 75° to 90° and drives procedural dual-arm running swings.
* **Camera Kinematics**: First-person head bobbing tied to physical distance traveled, Dutch roll/camera tilting (±4°) during lateral leaps and wall push-offs, and mouse pitch clamping [-89°, +89°].

### 1.2 Variable Jump System (`Spacebar`)
* **Dynamic Impulse**: Tapping `Spacebar` applies an initial vertical velocity impulse (v_y = 5.2 m/s).
* **Variable Hang Time**: Holding `Spacebar` taps into an active vertical lift reservoir for up to 0.25 seconds, enabling precision low hops on tap and high leaps on sustained hold.
* **Running Long Jump**: Horizontal sprint momentum converts directly into airborne trajectory distance.

### 1.3 Contextual Targeting Reticle & Visual Reach
* **Soft-Lock Reticle**: A non-pinpoint, magnetic HUD reticle tracks the screen-space projection of the nearest valid ledge rail within a 35° forward cone. 
* **Dynamic Sizing**: The reticle expands when out of range, tightens and highlights upon entering reach distance (<= 1.8 m), indicating exactly where the character's hands will snap if an interaction key is engaged.

### 1.4 Procedural Crevasses & Grip Anchors
Every obstacle procedurally defines and visually highlights three distinct grip heights:
* **Head-Height Crevasses (~2.0 m)**: Grabbable from standing or running ground height.
* **Jump-Height Crevasses (~3.8 m)**: Grabbable at the apex of a running jump.
* **Mantle Tops**: The flat, upper boundary of any wall or rooftop obstacle.

### 1.5 Multi-Key Parkour Combos & Dynamic Traversal
When targeted near a ledge rail:
* **Grab & Anchor (`E` / Buffered Grab)**: Anchors hands to the rail segment, zeroes velocity, and locks gravity.
* **Shimmy Left / Right (`A + E` / `D + E`)**: Traverses along the ledge tangent. Hands physically alternate grips along the rail. Movement is strictly clamped within the physical obstacle boundaries.
* **Climb Up / Mantle (`W + E`)**:
  * *On Lower Crevasse*: Smoothly hoists grip to the next vertical crevasse.
  * *At Top Lip*: Triggers an automated, two-stage **Mantle Vault** (vertical hoist followed by a forward step-through onto the top surface).
* **Climb Down / Drop (`S + E`)**: Steps down to a lower crevasse if present, or cleanly releases grip.
* **Dyno Jump Up (`Space + W + E`)**: Explosive vertical wall push (v_y = 8.5 m/s), hands push off the surface and enter an auto-reach state for higher out-of-reach ledges.
* **Wall Eject / Back Leap (`Space + S + E`)**: Explosive outward push along the wall normal vector (v_horiz = 7.5 m/s, v_y = 4.5 m/s), launching the player across gaps toward obstacles behind them.
* **Lateral Dyno Leaps (`Space + A + E` / `Space + D + E`)**: Launches sideways along the ledge tangent in a rising arc to clear gaps between adjacent pillars.

### 1.6 Fully Visual Procedural Arms, Hands, and Fingers
* **Analytical 2-Bone IK**: Solves shoulder, elbow, and wrist joints in real time to touch target ledge anchor points.
* **Procedural Finger Rig**: Hands accurately conform to geometry:
  * *Crimp Grip*: Fingers curl tightly over thin crevasse lips; thumbs brace against the wall face.
  * *Mantle Palm-Press*: Palms flatten against top horizontal surfaces; fingers splay forward.
  * *In-Flight Reach*: Hands anticipate oncoming edges as the player nears a valid target.

---

## 2. Technical Stack & Dependencies

* **Python 3.10+**
* **Pygame**: Window creation, input event dispatch, mouse locking, and high-resolution frame clocks.
* **PyOpenGL**: Modern OpenGL 3.3 Core Profile pipeline (VAOs, VBOs, EBOs, and GLSL shaders).
* **NumPy**: High-performance vectorized matrix algebra, Quaternion rotations, analytical IK equations, and mesh buffer staging.

---

## 3. High-Level Engine Architecture

The engine decouples physical simulation, input buffering, and skeletal IK from rendering:

```
┌────────────────────────────────────────────────────────┐
│                      Input Manager                     │
│  (Window events, mouse deltas, 150ms Action Queue)     │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│          Kinematic Character Controller (KCC)          │
│   • Swept AABB Collision      • Variable Jump Reservoir│
│   • Ground / Sprint Friction  • Air Momentum Physics   │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│            Parkour State Machine (FSM)                 │
│   • GROUND / SPRINT           • AIRBORNE (Jump/Fall)   │
│   • LEDGE_HANG / SHIMMY       • MANTLE_VAULT (Spline)  │
│   • DYNO_LAUNCH / WALL_EJECT  • Action Lockouts        │
└─────────────────────┬───────────────────┬──────────────┘
                      │                   │
┌─────────────────────▼───────┐   ┌───────▼──────────────┐
│  Parametric Ledge Graph     │   │ Soft-Lock Reticle    │
│  • Procedural Rail Extraction│   │ • Screen-space query │
│  • Segment Projections      │   │ • Magnetic smoothing │
└─────────────────────┬───────┘   └───────┬──────────────┘
                      │                   │
┌─────────────────────▼───────────────────▼──────────────┐
│      Procedural Skeletal Arm / Hand / Finger Rig       │
│   • Clavicle / Shoulder Camera Anchor with Inertia     │
│   • Analytical 2-Bone Arm IK (Shoulder-Elbow-Wrist)    │
│   • Procedural Hand Posing & Finger Crimp/Palm Solvers │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│           Modern OpenGL 3.3 Core Renderer              │
│   • Shader Programs (World, Ledge Glow, Skinned Arms)  │
│   • Dynamic Batched Geometry (VAO/VBO/EBO)             │
│   • Dynamic Camera (FOV Lerp, Bob, Roll, View Matrix)  │
└────────────────────────────────────────────────────────┘
```

---

## 4. Directory Structure

```
hardcore/
├── README.md                   # Project overview and controls
├── DESIGN.md                   # Full mathematical & architectural specification
├── main.py                     # Game loop, delta time, system orchestration
├── config.py                   # Global physics, IK, and rendering constants
├── core/
│   ├── window.py               # Pygame initialization & OpenGL 3.3 Core context
│   └── input_buffer.py         # Timestamped input queue and combo evaluator
├── math3d/
│   ├── transform.py            # NumPy 4x4 matrix helpers (Perspective, LookAt, Model)
│   ├── quaternion.py           # Rotations, SLERP, and direction-to-quaternion math
│   └── geometry.py             # Raycasts, segment projections, and distance tests
├── physics/
│   ├── aabb.py                 # Axis-Aligned Bounding Box primitives
│   └── collision.py            # Swept AABB vs static obstacle collision resolution
├── parkour/
│   ├── ledge_rail.py           # Parametric ledge segment data structures
│   ├── ledge_system.py         # Spatial queries, rail classification, and snap solver
│   └── state_machine.py        # KCC FSM (Ground, Hang, Shimmy, Mantle, Dyno)
├── animation/
│   ├── skeleton.py             # Joint hierarchies, local/world transforms, bone matrices
│   ├── two_bone_ik.py          # Analytical 2-bone arm solver with pole vectors
│   ├── finger_solver.py        # Procedural finger curling and surface contact logic
│   └── arm_rig.py              # First-person dual-arm system coordinator
├── world/
│   ├── obstacle.py             # Procedural parkour blocks, pillars, and walls
│   └── generator.py            # Procedural parkour course builder
└── render/
    ├── shader.py               # GLSL compile, link, and uniform caching utility
    ├── buffer.py               # Modern GL VAO/VBO/EBO abstractions
    ├── reticle.py              # Soft-lock contextual cursor renderer
    ├── renderer.py             # Master render pass (world, ledges, arms, HUD)
    └── shaders/
        ├── world.vert / .frag  # Scene geometry with directional lighting & grid lines
        ├── ledge.vert / .frag  # Glowing emissive rail highlight shader
        ├── arm.vert / .frag    # Skinned matrix palette shader for arms & fingers
        └── ui.vert / .frag     # Screen-space cursor shader
```

---

## 5. Controls Reference

| Input | Context | Action |
| :--- | :--- | :--- |
| **`W, A, S, D`** | Ground / Air | Directional movement & air steering |
| **`Left Shift` (Hold)** | Ground | Sprint (5.0 m/s -> 9.5 m/s, FOV expansion) |
| **`Spacebar` (Tap)** | Ground | Low vertical jump hop |
| **`Spacebar` (Hold <= 0.25s)** | Ground | High vertical leap / running long jump |
| **`E` (Tap / Hold)** | Near Rail | Contextual grab & anchor to ledge rail |
| **`A + E` / `D + E`** | Ledge Hang | Shimmy left / right along ledge rail |
| **`W + E`** | Ledge Hang | Climb to higher crevasse OR initiate Mantle Vault onto top surface |
| **`S + E`** | Ledge Hang | Climb down to lower crevasse OR drop from ledge |
| **`Space + W + E`** | Ledge Hang | Dyno Jump Up (8.5 m/s upward wall push) |
| **`Space + S + E`** | Ledge Hang | Wall Eject (7.5 m/s outward, 4.5 m/s upward) |
| **`Space + A/D + E`** | Ledge Hang | Lateral Dyno Leap across gaps |
| **`Mouse`** | Any | First-person look (Yaw & Pitch [-89°, +89°]) |
| **`Escape`** | Any | Unlock mouse / exit |
