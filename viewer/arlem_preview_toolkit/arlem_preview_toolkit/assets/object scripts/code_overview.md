# Object Scripts — Quick Reference

| Script | Category | Purpose |
|--------|----------|---------|
| `sortingActivity.cs` | Activity | Drag-and-sort interactive exercise: scrambles objects, player drags them into correct order, gives audio/visual feedback |
| `MoveToActivity.cs` | Activity | Move-to-target exercise: objects must be moved to specified target positions, checks order on button press |
| `matchingActivity.cs` | Activity | Pair-matching exercise: player drags objects to match paired items; incomplete/in-progress |
| `moveObjects.cs` | Motion | Animates a GameObject through start→mid→final position/rotation/scale over a time window using quadratic path fitting |
| `generalOrbit.cs` | Motion | Orbits an object around a named parent with controllable start/end time, time rate, and synchronous (tidal lock) rotation |
| `simpleOrbit.cs` | Motion | Simpler continuous orbit (no start/end time control); always-running version of generalOrbit |
| `orbit.cs` (class: `revolve`) | Motion | Minimal orbit using parent scale; radius and speed only — no angle/phase control |
| `generalRotation.cs` | Motion | Rotates an object around Y axis with start/end time window and programmable target angle |
| `simpleRotation.cs` | Motion | Minimal always-running Y-axis rotation — no time window |
| `SimpleZRotation.cs` | Motion | Continuous Z-axis spin at a fixed rate — one-liner utility |
| `monthlyMotion.cs` | Motion | Drives Earth-Moon-Sun system simulation with UI buttons for time stepping (MagicLeap InputReceiver) |
| `monthlyMotionMod.cs` | Motion | Simplified version of monthlyMotion with no UI buttons; auto-advances and displays a TextMeshPro day counter |
| `lightingControl.cs` | Scene | Singleton: saves, restores, or replaces scene directional light with a sunlight configuration |
| `checkAngle.cs` | Interaction | On drag-end, measures azimuth/altitude angle of this object relative to a "Sun" object and checks against a target range; plays audio and calls back on success |
| `buttonCallback.cs` | Interaction | MagicLeap InputReceiver button that calls `sortingActivity.feedbackOnOrder()` on click or drag-end |
| `demoButtonActions.cs` | Interaction | MagicLeap InputReceiver button that calls `demoSequence.actionCallBack()` on a list of named objects |
| `finalieButtonCallback.cs` | Interaction | MagicLeap InputReceiver button intended to signal lab completion; body is stubbed out (call commented) |
| `finalScreen.cs` | Scene | Sets a final-slide texture, colors end buttons red, adds `finalieButtonCallback`, and plays a final audio clip |
| `placeObjects.cs` | Utility | Pure C# class (not MonoBehaviour): generates arrays of 3D positions in grid, ring, concentric-rings, or line layouts |
| `transformTracker.cs` | Debug | Logs position/localPosition every 5 seconds to LabLogger — attach to any object to track it |
| `DontDestroyOnLoad.cs` | Utility | Calls `DontDestroyOnLoad(gameObject)` in Awake — keeps the object alive across scene loads |
| `FollowTransform.cs` | Utility | Copies another transform's world position each frame; start/pause/stop API |
| `HeadposeCanvas.cs` | Utility (Magic Leap) | Lerps a Canvas toward the camera forward each frame; keeps UI in the headset field of view |
| `sortingActivityData.cs` | Data | Serializable data container for sortingActivity: object list, audio clips, flags — no behavior |
| `MoveToData.cs` (class: `MoveToActivityData`) | Data | Serializable data container for MoveToActivity: audio clips, flags, timing — no behavior |
| `sortingActivityOld.cs` | Legacy | Earlier standalone version of sortingActivity (not ActivityModule subclass); hardcoded Inspector fields instead of JSON data — superseded |

**Redundant / non-behavior scripts:** `sortingActivityData.cs`, `MoveToData.cs` — pure data containers, no logic.  
`DontDestroyOnLoad.cs`, `FollowTransform.cs`, `HeadposeCanvas.cs` — duplicated from the `unity code/` directory; identical copies.  
`sortingActivityOld.cs` — superseded by `sortingActivity.cs`; kept for reference only, not for production use.  
`finalieButtonCallback.cs` — the key callback body is commented out; effectively a stub.  
`orbit.cs` (class name `revolve`) — filename/class name mismatch; minimal functionality covered by `simpleOrbit.cs`.  
`matchingActivity.cs` — compiles but is incomplete; `matchNames` dictionary is never initialized and will throw `NullReferenceException` at runtime.

**Motion script family:** `simpleOrbit` → `generalOrbit` (adds time windowing) → `monthlyMotion` (full Earth-Moon-Sun system). Similarly `simpleRotation` → `generalRotation` (adds time windowing and target-angle control) → `SimpleZRotation` (Z-axis only, no controls).

---

## Detailed Reference

### `sortingActivity.cs`
**Category:** Activity (namespace `sortingRoutines`)  
**Inherits:** `ActivityModule`

The main interactive sorting exercise. On `Initialize()` it deserializes `sortingActivityData` from JSON, spawns AR objects via `Bridge`, plays intro audio, and optionally sets sunlight. In `Start()` it builds a `sortInfo[]` array pairing each AR object with its correct target position, places grey sphere markers at the target spots, attaches `moveObjects` to each sortable, then scrambles them with `resetPositions()`.

The player drags objects with the MagicLeap controller. Each `OnDragEnd` fires `resort()`, which finds the most displaced object and re-sorts the visual order. The big red button triggers `feedbackOnOrder()`.

**Feedback loop:**
- Correct order → green markers, success audio, 5s delay, destroy objects/markers/button, call `LabManager.ModuleComplete()`
- Wrong order → red/green markers, wrong-answer audio (cycles through pool), scramble and re-sort

**Key inspector fields:** `wrongOrder[5]` (AudioClip array), `correctOrder` (AudioClip), `myButton` (GameObject)

**Key methods:**
| Method | Description |
|--------|-------------|
| `Initialize(string jsonData)` | Deserializes data, sets up scene, starts module |
| `EndOfModule()` | Calls `taskCompleted()` coroutine to clean up and advance |
| `feedbackOnOrder()` | Checks order, plays audio, updates lights |
| `resort()` | Called on drag-end; re-evaluates object positions and re-sorts |
| `createMarkers(Transform, float, float)` | Instantiates grey tinysphere markers at target positions |

**Note:** `findDisplacedObject()` inner loop runs twice identically — appears to be a copy-paste bug with no functional effect.

---

### `MoveToActivity.cs`
**Category:** Activity (namespace `MoveToRoutines`)  
**Inherits:** `ActivityModule`

Functionally nearly identical to `sortingActivity`. Differences: uses `MoveToActivityData` instead of `sortingActivityData`, parses object positions from an `ObjectInfoCollection` JSON, and uses `Bridge.ParseJson()` instead of `Bridge.makeObjects()`. The sorting and movement logic is the same.

**Key inspector fields:** `wrongOrder[5]` (AudioClip array), `correctOrder` (AudioClip)

**Key methods:** same interface as `sortingActivity` — `Initialize`, `EndOfModule`, `SaveState`, `feedbackOnOrder`, `resort`

**Note:** `taskCompleted()` finds `"Lab Control"` but the call to its completion method is commented out — module advancement is broken at that point.

---

### `matchingActivity.cs`
**Category:** Activity — **incomplete** (namespace `sortingRoutines`)  
**Inherits:** `ActivityModule`

Intended for a pair-matching exercise where objects are matched to their named partner. `Initialize()` and the `matchInfo` inner class are in place, but `matchNames` (a `Dictionary<string,string>`) is never `new`-ed; `Start()` will throw a `NullReferenceException` at the `matchNames.Add(...)` call.  

The `resetPositions()` and feedback methods present in `sortingActivity` are absent — the activity cannot complete. **Do not attach to production prefabs without finishing the implementation.**

---

### `moveObjects.cs`
**Category:** Motion  
**Inherits:** `MonoBehaviour`

Animates a GameObject through a three-point path (start → mid → final) over a specified `TimeRange`. Position uses a quadratic Bézier-style fit via `threeVectorQuadratic()`; rotation and scale use linear interpolation via `twoVectorLinear()`. The animation runs only when `Time.time` falls within `TimeRange` — set `TimeRange` to negative values to disable.

**Properties (get/set):**

| Property | Type | Description |
|----------|------|-------------|
| `StartPos` / `MidPos` / `FinalPos` | Vector3 | Path waypoints |
| `StartSize` / `FinalSize` | Vector3 | Scale endpoints |
| `StartAngle` / `FinalAngle` | Vector3 | Euler angle endpoints |
| `TimeRange` | Vector2 | `[start, end]` in game-time seconds |

**Key methods:**
| Method | Description |
|--------|-------------|
| `initializePath()` | Recomputes polynomial coefficients — must call after changing properties |

**Usage note:** All three waypoints must be set, then `initializePath()` called, before the animation will work. See the comment block at the top of the file for a complete usage example.

---

### `generalOrbit.cs`
**Category:** Motion  
**Inherits:** `MonoBehaviour`

Orbits this object around a named planet in the XZ plane. Distance is computed from the initial `moonPosition` relative to the planet's world position at Start. Supports an active time window (`orbitStartTime`/`orbitEndTime`) and a `timeRate` multiplier.

**Inspector fields:**

| Field | Type | Description |
|-------|------|-------------|
| `moonPosition` | Vector3 | Starting world position of the orbiting body |
| `nameOfPlanetBeingOrbited` | string | Scene name of the center object |
| `orbitalPeriod` | float | Seconds per full orbit |
| `orbitScale` | float | Multiplier on the orbital radius |
| `synchronousRotation` | bool | If true, same face always points at planet (tidal lock) |
| `orbitStartTime` / `orbitEndTime` | float | Active time window (game seconds) |
| `timeRate` | float | Speed multiplier |

**Key methods:**
| Method | Description |
|--------|-------------|
| `runForSpecifiedTime(float interval, float delay)` | Run for `interval` seconds after `delay` |
| `moveToAngle(float thetaFinal, float delay)` | Animate to a specific orbital angle |
| `findTimeToAngle(float theta)` | Returns time needed to reach an angle |
| `updateSimulation(float tstart, float tend, float rate)` | Direct time-window override |

---

### `simpleOrbit.cs`
**Category:** Motion  
**Inherits:** `MonoBehaviour`

Always-running orbital motion (no time window). Uses `Time.time` directly so the orbit never stops. Otherwise structurally identical to `generalOrbit`. Prefer `generalOrbit` when you need scripted start/stop control.

**Inspector fields:** `moonPosition`, `nameOfPlanetBeingOrbited`, `orbitalPeriod`, `timeRate`, `orbitScale`, `synchronousRotation`

---

### `orbit.cs` (class: `revolve`)
**Category:** Motion  
**Inherits:** `MonoBehaviour`

**Filename/class name mismatch** — the file is `orbit.cs` but the class is `revolve`.

Minimal orbit: accumulates `_angle` each frame and sets world position relative to the parent's position. Does not target a named object by string — relies on `transform.parent`. Scale-aware (uses parent's X and Z scale as ellipse radii).

**Inspector fields:** `RotateSpeed` (float, radians/s), `Radius` (float)

Use `simpleOrbit` or `generalOrbit` for anything beyond a simple visual spinner.

---

### `generalRotation.cs`
**Category:** Motion  
**Inherits:** `MonoBehaviour`

Y-axis rotation with a programmable active time window. Sets `transform.eulerAngles` directly each frame (not `Rotate()`), so it fights any other script setting euler angles.

**Inspector fields:** `rotationTime` (seconds per revolution), `rotationStartTime`, `rotationEndTime`, `timeRate`

**Key methods:**
| Method | Description |
|--------|-------------|
| `rotateForSpecifiedTime(float interval, float delay)` | Rotate for a fixed duration |
| `moveToAngle(float thetaFinal, float delay)` | Animate to a target angle |
| `findTimeToAngle(float theta)` | Time to reach a given angle |
| `updateSimulation(float tstart, float tend, float rate)` | Direct time-window override |

**Known bug:** `rotationAngle` is accumulated as `rotationAngle - rotationRate * Time.time * timeRate` rather than `* Time.deltaTime`, so rotation rate is dependent on absolute game time rather than elapsed time — the object accelerates indefinitely. This is likely a copy error from an earlier version.

---

### `simpleRotation.cs`
**Category:** Motion  
**Inherits:** `MonoBehaviour`

Minimal always-running Y-axis rotation. Sets `localEulerAngles` proportional to `Time.time * timeRate` each frame — smooth, frame-rate independent, but not stoppable.

**Inspector fields:** `timeRate` (multiplier), `rotationTime` (seconds per revolution)

---

### `SimpleZRotation.cs`
**Category:** Motion  
**Inherits:** `MonoBehaviour`

Single-purpose: rotates the object around the Z axis every frame using `transform.Rotate()`. Good for spinner/loading indicators. Default rate is 3/4 turn per second.

**Inspector field:** `rotSpeed` (float, radians/s, SerializeField)

---

### `monthlyMotion.cs`
**Category:** Motion  
**Inherits:** `MonoBehaviour`

Drives a multi-body Earth-Moon-Sun simulation. Requires six scene GameObjects to be wired in: `theEarth`, `theMoon`, `earthCenter`, `moonCenter`, `earthStarBeam`, plus UI buttons. Advances a `theTime` variable (in days) each frame by `timeRate`, then sets euler angles on each body to match their orbital positions.

UI buttons (MagicLeap `InputReceiver.OnClick`) wire to step-forward/back (+5 days, +6 hours) and run/stop controls.

**Note:** `onEnable()` / `onDisable()` are lowercase — they are NOT Unity callbacks. `Start()` calls `onEnable()` manually. `OnEnable()` / `OnDisable()` (capitalized) are never called, so the listeners are registered once and never removed on scene reload.

**Inspector fields:** All six body GameObjects, six button GameObjects, `textObject`, `timeRate`, `systemRotationTime`, `moonRotationTime`, `earthRotationTime`

---

### `monthlyMotionMod.cs`
**Category:** Motion  
**Inherits:** `MonoBehaviour`

Simplified version of `monthlyMotion` with no UI buttons — the simulation just runs automatically. Adds a `TextMeshPro` day counter. Same body GameObjects required.

Use this when you want the planetary animation to play passively without user time-stepping controls.

**Inspector fields:** same body GameObjects as `monthlyMotion`, `textObject` (TextMeshPro), `timeRate`, rotation time fields

---

### `lightingControl.cs`
**Category:** Scene  
**Inherits:** `MonoBehaviour` (Singleton)

Controls the scene's Directional Light. On Awake it finds the "Directional Light" object and saves its current state. Can switch to a fixed "sunlight" configuration or restore the original.

**Singleton access:** `lightingControl.Instance`

**Key methods:**
| Method | Description |
|--------|-------------|
| `saveLights()` | Captures current light transform, intensity, type, color, shadows |
| `restoreLights()` | Restores the saved state |
| `sunlight()` | Applies hardcoded sunlight config (directional, white, no shadows, intensity 2.5) |
| `setSunlight(Vector3, Quaternion, Vector3, float)` | Override the default sunlight parameters |

---

### `checkAngle.cs`
**Category:** Interaction  
**Inherits:** `MonoBehaviour`

On MagicLeap pointer drag-end, computes the azimuth and altitude of this object relative to the camera, compares against a target angle with tolerance, and logs the result to `LabLogger`. On success, plays an audio clip from `MediaCatalogue` then calls `demoSequence.actionCallBack()` on each named callback object.

**Inspector fields:**

| Field | Type | Description |
|-------|------|-------------|
| `azmTarget` / `azmTolerance` | float | Azimuth target (degrees) and allowed error |
| `altTarget` / `altTolerance` | float | Altitude target (degrees) and allowed error |
| `audioClipSuccess` | string | MediaCatalogue key for success sound |
| `callBackObjects` | string[] | Names of GameObjects to notify on success |

**Hardcoded dependency:** looks for scene objects named `"Sun"` and `"Main Camera"` by name.

---

### `buttonCallback.cs`
**Category:** Interaction  
**Inherits:** `MonoBehaviour`

Wires a MagicLeap `InputReceiver.OnSelected` (and always `OnDragEnd`) to call `sortingActivity.feedbackOnOrder()` on the "sortingManager" GameObject. The `enableOnClick` flag gates whether the `OnSelected` path registers.

**Inspector field:** `enableOnClick` (bool, default true)

**Note:** Finds "sortingManager" by name at runtime — `sortingActivity.Start()` renames its host to this name.

---

### `demoButtonActions.cs`
**Category:** Interaction  
**Inherits:** `MonoBehaviour`

A generic button that calls `demoSequence.actionCallBack(gameObject)` on each of a list of named GameObjects when clicked via MagicLeap `InputReceiver.OnSelected`.

**Inspector field:** `callBackObjects` (string[]) — names of scene objects to notify

---

### `finalieButtonCallback.cs`
**Category:** Interaction — **stub**  
**Inherits:** `MonoBehaviour`

MagicLeap `InputReceiver` button that was intended to call `LabControl.finalieDone()` when clicked or drag-ended. That call is commented out. The handler finds "Lab Control" but does nothing. Effectively non-functional as written.

---

### `finalScreen.cs`
**Category:** Scene  
**Inherits:** `MonoBehaviour`

Wires up the end-of-lab screen in `Start()`: sets the object's renderer texture to `finalSlide`, colors two end buttons red, programmatically adds a `finalieButtonCallback` component to the "endApp" object, and plays `finalAudio`.

**Inspector fields:** `finalSlide` (Texture), `finalAudio` (AudioClip)

**Hardcoded dependencies:** searches for "endApp" and "endbutton" by name.

---

### `placeObjects.cs`
**Category:** Utility  
**Pure C# class — not a MonoBehaviour**

Layout helper that generates arrays of `Vector3` positions. Constructed with object count, height, and width; then call one of the layout methods to fill `ptLocation[]`.

**Constructor:** `placeObjects(int nObjects, float height, float width)`

**Layout methods:**
| Method | Description |
|--------|-------------|
| `createLine()` | Evenly spaced in a horizontal line |
| `createRing(bool horizontal)` | Elliptical ring; horizontal = XZ plane, else XY |
| `createRings(int nrings)` | Concentric rings, proportional distribution per ring |
| `createGrid()` | Rectangular grid, top-left origin |
| `createOddGrid()` | Grid with the last partial row centered |
| `transformGrid(...)` | Rotates and offsets the entire position array |
| `nearestObject(Vector3)` | Returns (index, distance) of nearest point to a position |

**Note:** A second `static` version of the class is commented out at the bottom of the file. The active instance-based version is what should be used.

---

### `transformTracker.cs`
**Category:** Debug  
**Inherits:** `MonoBehaviour`

Attaches a coroutine in `Start()` that logs this object's world and local position to `LabLogger` every 5 seconds. Useful for diagnosing transform issues during development. Has no inspector fields and no effect on game behavior — safe to attach and remove without side effects.

---

### `sortingActivityData.cs`
**Category:** Data  
**Serializable class (namespace `sortingRoutines`)  — no behavior**

Data container for `sortingActivity.Initialize()`. Extends `ActivityModuleData`.

**Fields:** `objects` (ObjectInfo[]), `introAudio` (string), `useSunlight` (bool), `timeToEnd` (float, -1 = never), `endUsingButton` (bool), `createObjects` / `destroyObjects` / `restoreLights` (bool), `wrongOrderAudio` / `correctOrderAudio` (string)

---

### `MoveToData.cs` (class: `MoveToActivityData`)
**Category:** Data  
**Serializable class (namespace `MoveToRoutines`) — no behavior**

Data container for `MoveToActivity.Initialize()`. Extends `ActivityModuleData`.

**Fields:** same pattern as `sortingActivityData` but `objects` is a `string` (JSON blob) rather than `ObjectInfo[]`.

---

### `sortingActivityOld.cs`
**Category:** Legacy — superseded  
**Inherits:** `MonoBehaviour` (not `ActivityModule`)

The predecessor to `sortingActivity`. Differences from the current version:
- Does not extend `ActivityModule` — cannot be used with `LabManager`
- Object count determined by counting non-null Inspector-assigned `Texture[]` slots, not JSON
- Creates its own sortable panels with `createSortables()` using a prefab + texture array
- Uses hardcoded sort-line geometry parameters instead of reading target positions from JSON
- `taskCompleted()` looks for `"Lab Control"` but the completion call is commented out

**Do not use in new modules.** Retained as a reference for how the sorting algorithm was originally implemented.

---

### `DontDestroyOnLoad.cs` / `FollowTransform.cs` / `HeadposeCanvas.cs`
These three files are **identical copies** of the scripts in the `unity code/` sibling directory. See [code_overview.md in the unity code directory](../unity%20code/code_overview.md) for full documentation. No behavioral differences.
