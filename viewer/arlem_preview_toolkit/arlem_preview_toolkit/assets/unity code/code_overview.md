# Unity C# Scripts — Quick Reference

| Script | Category | Purpose |
|--------|----------|---------|
| `LabManager.cs` | Core | Loads and runs an AR lab: connects to Photon Fusion, spawns activity modules, manages colocation anchor, handles module transitions |
| `LoginManager.cs` | Core | Login/lab-selection state machine: downloads lab list, PIN auth, generates UI, loads lab scene |
| `Bridge.cs` | Core | Singleton gateway for spawning/modifying networked AR objects via Photon Fusion (or local Resources fallback) |
| `MediaPlayer.cs` | Core | Singleton media controller for audio, video, and image assets; manages playback state and slider sync |
| `labPlacementHandling.cs` | AR | XR trigger-based AR anchor placement using ARFoundation `ARAnchorManager` |
| `ActivityModule.cs` | Base Class | Abstract MonoBehaviour base that all activity modules extend; defines lifecycle interface |
| `FollowTransform.cs` | Utility | Makes a GameObject copy another transform's position each frame; start/pause/stop API |
| `HeadposeCanvas.cs` | Utility | Lerps a Canvas toward the camera's forward direction (Magic Leap head-pose billboard) |
| `LogThisObject.cs` | Utility | Registers/unregisters a GameObject with LabLogger on Start/OnDestroy |
| `DontDestroyOnLoad.cs` | Utility | Calls `DontDestroyOnLoad(gameObject)` in Awake — one-liner scene-persistence helper |
| `ObjectInfo.cs` | Data | Serializable AR object specification: prefab name, position, rotation, scale, physics, text, etc. |
| `LabDataObject.cs` | Data | Top-level lab JSON container: Lab_ID, Author, CourseName, ActivityModules, Assets, Transmission flag |
| `ActivityModuleData.cs` | Data | Serializable module data: name, prefab, objectives, instructions, media IDs, grading fields |
| `ComponentName.cs` | Data Helper | Single-field wrapper (`string name`) used by Bridge to pass component names; no independent behavior |
| `ModuleTemplate.cs` | Template | Copy-paste starter for new ActivityModule implementations; not production code |
| `ModuleTemplateData.cs` | Template | Empty subclass of ActivityModuleData; exists only as a companion stub for ModuleTemplate |

**Redundant / non-behavior scripts:** `ComponentName.cs`, `ModuleTemplate.cs`, `ModuleTemplateData.cs` — none add runtime behavior. `ActivityModuleData.cs` and `LabDataObject.cs` are data containers with no logic but are required by the serialization chain.

**Scripts with significant disabled code:** `Bridge.cs` (Transmission multiplayer system replaced by Photon Fusion — ~300 lines commented out), `LabManager.cs` (server download replaced by local file load — several blocks commented out).

---

## Detailed Reference

### `LabManager.cs`
**Category:** Core behavior  
**Inherits:** `SimulationBehaviour, IPlayerJoined` (Photon Fusion)

Orchestrates the full lab runtime. On start it connects to a Fusion `NetworkRunner` session, reads the lab zip from `StreamingAssets`, deserializes `LabDataObject` JSON (via `JsonUtility`), initializes a `MediaCatalogue`, and sequentially instantiates `ActivityModule` prefabs for each `ActivityModuleData` entry.

Colocation anchor: listens for two controller button clicks in sequence to place the shared AR anchor via `LabManager.SetSharedAnchor()`.

`IPlayerJoined.PlayerJoined()` is called by Fusion when a remote peer connects; re-runs module initialization for the new player.

**Key methods:**
| Method | Description |
|--------|-------------|
| `Start()` | Connects Fusion runner, loads lab data |
| `InitializeModules()` | Instantiates and calls `Initialize()` on each ActivityModule prefab |
| `ModuleComplete()` | Called by the current module when finished; advances to next module |
| `PlayerJoined(PlayerRef)` | Fusion callback — syncs new player into current module state |
| `SetSharedAnchor()` | Places colocation anchor for multi-user alignment |

**Inspector fields:** `labFileName` (string), `networkRunnerPrefab`, `activityModulePrefabs[]`, `mediaPlayer`, `anchorManager`

---

### `LoginManager.cs`
**Category:** Core behavior  
**Inherits:** `MonoBehaviour`

Four-state machine controlling the pre-lab flow:

| State | Behavior |
|-------|---------|
| `STARTUP` | Initialization, transitions immediately |
| `INTRODUCTION` | Shows intro screen |
| `LOGIN` | Displays PIN pad; validates against server |
| `LAB_SELECTION` | Downloads lab manifest from `https://cyberlearnar.cs.mtsu.edu`, generates one button per lab, loads lab scene on selection |

PIN auth sends credentials to the cyberlearnar server and awaits a response before allowing lab access. Lab selection dynamically instantiates UI buttons at runtime.

**Key methods:** `Update()` (state dispatch), `OnPinSubmitted(string)`, `LoadLabList()` (coroutine), `OnLabSelected(string)`

---

### `Bridge.cs`
**Category:** Core behavior  
**Inherits:** None (plain C# Singleton — not a MonoBehaviour)

Central factory for AR object creation and modification. Checks `transmissionEnabled` on the lab data to decide whether to spawn via Fusion `NetworkRunner.Spawn()` (networked) or `Resources.Load` + `Instantiate` (local).

The original Transmission multiplayer backend is fully commented out. The active path uses Photon Fusion exclusively.

**Key methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `MakeObject` | `(ObjectInfo) → GameObject` | Spawns a single AR object |
| `MakeObjects` | `(List<ObjectInfo>)` | Batch-spawns a list |
| `ModifyNetworkObject` | `(string id, ObjectInfo)` | Updates an existing networked object's transform/components |
| `CleanUp` | `()` | Despawns all tracked network objects |

**Singleton access:** `Bridge.Instance`

---

### `MediaPlayer.cs`
**Category:** Core behavior  
**Inherits:** `MonoBehaviour` (Singleton)

Manages playback of audio, video, and image assets identified by a `MediaType` enum (`Audio`, `Video`, `Image`). Maintains a slider UI that syncs with audio/video position in `Update()`.

**Key methods:**
| Method | Description |
|--------|-------------|
| `Play(string mediaID, MediaType)` | Loads and starts playback |
| `Pause()` / `Restart()` / `Skip()` | Playback controls |
| `handleOnSliderValueChanged(float)` | Slider callback — seeks to position |

**Singleton access:** `MediaPlayer.Instance`

---

### `labPlacementHandling.cs`
**Category:** AR behavior  
**Inherits:** `MonoBehaviour`

Handles the initial physical placement of the AR lab anchor in the real world. Polls XR Input for a trigger press, then calls `ARAnchorManager.TryAddAnchorAsync()` at the controller's current pose. Once placed, `_placed` is set to `true` and the anchor GameObject is stored in `_anchor`.

In `Update()`, before placement the script follows the controller's world position so the preview moves with the hand.

**Public fields:** `_placed` (bool), `_anchor` (ARAnchor) — read by LabManager to retrieve the anchor pose.  
**Inspector fields:** `anchorManager` (ARAnchorManager), `controllerTransform` (Transform)

---

### `ActivityModule.cs`
**Category:** Base class  
**Inherits:** `MonoBehaviour` (abstract)

Defines the interface all activity modules must implement. LabManager instantiates prefabs that have a concrete subclass attached and calls `Initialize()` to start the module.

**Abstract methods (must override):**
| Method | Signature | Description |
|--------|-----------|-------------|
| `Initialize` | `(string labID, int moduleIndex)` | Full init with lab context |
| `Initialize` | `(string labID)` | Minimal init overload |
| `EndOfModule` | `()` | Clean up and signal completion to LabManager |
| `SaveState` | `()` | Persist module progress |

---

### `FollowTransform.cs`
**Category:** Utility  
**Inherits:** `MonoBehaviour`

Copies a target transform's world position to this GameObject each frame while following is active. Useful for attaching UI or indicators to a tracked object without full parenting.

**Inspector field:** `target` (Transform)

**Key methods:**
| Method | Description |
|--------|-------------|
| `Follow(Transform)` | Sets target and enables position copy |
| `PauseFollow()` | Stops copying without clearing target |
| `ResumeFollow()` | Restarts copying with existing target |
| `StopFollow()` | Disables copying and clears target |

---

### `HeadposeCanvas.cs`
**Category:** Utility (Magic Leap)  
**Inherits:** `MonoBehaviour`

Billboard component adapted from the Magic Leap SDK. Lerps the Canvas GameObject toward the camera's forward direction every frame so UI stays in the user's field of view.

**Inspector fields:**
| Field | Type | Description |
|-------|------|-------------|
| `positionLerpSpeed` | float | How fast position follows camera |
| `rotationLerpSpeed` | float | How fast rotation follows camera |
| `poseOffset` | Vector3 | Offset from the camera forward point |
| `keepVertical` | bool | (Local edit) Locks Y-axis so canvas doesn't tilt with head pitch |

---

### `LogThisObject.cs`
**Category:** Utility  
**Inherits:** `MonoBehaviour`

One-purpose script: registers the attached GameObject with a `LabLogger` singleton in `Start()` and unregisters in `OnDestroy()`. Add to any object that should appear in the lab's activity log.

No public fields. Depends on `LabLogger` being present in the scene.

---

### `DontDestroyOnLoad.cs`
**Category:** Utility  
**Inherits:** `MonoBehaviour`

Calls `DontDestroyOnLoad(gameObject)` in `Awake()`. Attach to any GameObject that must persist across scene loads (e.g., managers, singletons). No configuration needed.

---

### `ObjectInfo.cs`
**Category:** Data  
**Serializable class (not MonoBehaviour)**

Primary specification for a single AR object to be placed in the scene. Serialized as part of `LabDataObject` JSON via `JsonUtility`.

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `objectID` | string | Unique identifier |
| `prefabName` | string | Name of prefab in Resources |
| `position` | Vector3 | World position |
| `rotation` | Quaternion | World rotation |
| `scale` | Vector3 | Local scale |
| `components` | List\<ComponentName\> | Component names to enable/configure |
| `objectText` | string | Text content (for TextMeshPro components) |
| `isNetworked` | bool | Whether to spawn via Fusion |

**Helper inner classes (also in this file):**
- `rigidBodyClass` — physics settings (mass, drag, useGravity)
- `pointerReceiverClass` — interaction receiver settings
- `textProClass` — TextMeshPro text/font configuration

---

### `LabDataObject.cs`
**Category:** Data  
**Serializable class (not MonoBehaviour)**

Top-level container deserialized from the lab ZIP's JSON file. Passed to LabManager and Bridge to drive the entire lab session.

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `Lab_ID` | string | Unique lab identifier |
| `Author` | string | Lab author name |
| `CourseName` | string | Course this lab belongs to |
| `ActivityModules` | List\<ActivityModuleData\> | Ordered list of modules |
| `Assets` | List\<ObjectInfo\> | Static AR assets for the scene |
| `Transmission` | bool | Whether multiplayer sync is active (drives Bridge.transmissionEnabled) |

---

### `ActivityModuleData.cs`
**Category:** Data  
**Serializable class (not MonoBehaviour)**

Data payload for a single activity module. Deserialized from JSON and passed to the corresponding `ActivityModule.Initialize()` call.

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `moduleName` | string | Display name |
| `prefabName` | string | Prefab in Resources to instantiate |
| `objectives` | string[] | Learning objectives |
| `instructions` | string[] | Step-by-step instructions |
| `mediaIDs` | string[] | Media asset IDs available in this module |
| `gradingEnabled` | bool | Whether grading logic runs |
| `passingScore` | float | Minimum score to pass (0–1) |

---

### `ComponentName.cs`
**Category:** Data Helper — no independent behavior  
**Serializable class (not MonoBehaviour)**

Single-field wrapper used by `Bridge` when passing component names to `ModifyNetworkObject`. Contains only `public string name`. No methods, no logic.

---

### `ModuleTemplate.cs`
**Category:** Template — not production code  
**Inherits:** `ActivityModule`

Copy-paste starter for building new activity modules. Contains stub overrides for all `ActivityModule` abstract methods with `// TODO` comments. Should be duplicated and renamed when creating a new module — **do not attach this script directly to production prefabs.**

---

### `ModuleTemplateData.cs`
**Category:** Template — not production code  
**Inherits:** `ActivityModuleData`

Empty companion data class for `ModuleTemplate`. Has no fields or methods beyond the inherited ones. Exists so the template pair compiles cleanly. Replace with a real data class when building a new module.
