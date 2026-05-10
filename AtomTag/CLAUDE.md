# AtomTag

An Android app that detects AprilTags through the device camera, computes their
6-DOF poses in a fixed field coordinate frame, broadcasts those poses over UDP
multicast, and provides a dashboard for monitoring and controlling the
robots/devices identified by those tags.

The app is a Compose-first, single-Activity app that runs detection in a
foreground `LifecycleService` so broadcast continues while the UI is in the
background or the camera modal is closed.

---

## Quick start

### Build / install on a connected device or running emulator

```bash
./gradlew installDebug
```

### Run from Android Studio

Use the standard ▶ Run button. The Gradle wrapper handles the rest.

### Open camera + permissions

On first run the app requests `CAMERA` and (on Android 13+) `POST_NOTIFICATIONS`.
The detection service starts only after camera permission is granted; without
it, the dashboard still loads but detection / broadcast does nothing.

### Inspect persistent user data on device

```bash
adb shell run-as com.atomtag cat files/device_overrides.json
```

---

## High-level architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ MainActivity (AppCompatActivity)                                   │
│   • TagConfig.load(this)         ← reads assets/tag_config.json    │
│   • UserPreferences.init(ctx)    ← reads files/device_overrides... │
│   • OpenCVLoader.initLocal()     ← loads native lib                │
│   • Permission handling (camera, notifications)                    │
│   • Starts + binds DetectionService                                │
│   • setContent { CompositionLocalProvider(...) { AppRoot() } }     │
└──────────────────────┬─────────────────────────────────────────────┘
                       │ binds
                       ▼
┌────────────────────────────────────────────────────────────────────┐
│ DetectionService (LifecycleService, foreground, type=camera)       │
│   • Owns: AprilTagDetector, PoseTransformer, UdpBroadcaster,       │
│     PoseVector, CameraX (Preview + ImageAnalysis use cases),       │
│     toggle state (axes/labels/virtualBg), CameraStats StateFlow    │
│   • Persistent notification with "Stop" action                     │
│   • attachPreview/detachPreview ← surface comes from the modal     │
│   • attachOverlay/detachOverlay ← AxisOverlayView from the modal   │
│   • analyzeFrame: detect → transformToFieldFrame → broadcast →     │
│     update overlay + stats                                         │
└──────────────────────┬─────────────────────────────────────────────┘
                       │ provided via LocalDetectionService CompositionLocal
                       ▼
┌────────────────────────────────────────────────────────────────────┐
│ AppRoot (Compose root)                                             │
│   • AtomTagTheme (forced dark)                                     │
│   • State-based screen swap: Dashboard <-> Settings                │
│   • ModalBottomSheet for camera, hosts CameraScreen                │
└──────────────┬───────────────────────────┬─────────────────────────┘
               │                           │
               ▼                           ▼
       DashboardScreen                  CameraScreen
       (devices list, mode panel,       (preview + overlay AndroidViews,
        FAB row)                        toggle chips, stats strip)
               │
               ├─ DeviceActionSheet (per-device sheet with rename, ping,
               │  restart, color picker, team selector, status messages)
               └─ ModeSelectorPanel (mode dropdown + Apply with ack
                  counter)
```

---

## Source tree

```
AtomTag/
├── app/
│   ├── build.gradle.kts                  ← deps, build variants, Compose options
│   └── src/main/
│       ├── AndroidManifest.xml           ← permissions, service decl, theme
│       ├── assets/tag_config.json        ← origin tag, multicast cfg, field
│       │                                    frame offset, per-tag size+label
│       ├── res/
│       │   ├── drawable/                 ← notification icon (vector)
│       │   ├── mipmap-*/                 ← launcher icons (adaptive)
│       │   └── values/                   ← app strings, launcher bg color
│       └── java/com/atomtag/
│           ├── MainActivity.kt           ← entry, init, perms, service binding
│           ├── data/
│           │   ├── AppMode.kt            ← enum (Sandbox/Teleop/Solo/Team)
│           │   │                            + DeviceHealth (Ok/Warn/Error)
│           │   ├── DeviceActions.kt      ← PingResult, RestartState
│           │   ├── DeviceState.kt        ← UI-facing per-device snapshot
│           │   ├── DeviceStateRepository ← interface + setters/applyMode
│           │   ├── MockDeviceStateRepo   ← simulated battery drift, mode acks
│           │   └── UserPreferences.kt    ← persistent name/color/team
│           ├── detection/
│           │   ├── AprilTagDetector.kt   ← ArUco detection, ROI re-detect,
│           │   │                            origin-pose EMA filter, 3D→2D
│           │   │                            axis projection (incl. field axes)
│           │   └── PoseTransformer.kt    ← cam→origin frame, origin→field
│           ├── model/
│           │   ├── DetectionResult.kt    ← per-tag detector output
│           │   ├── PoseVector.kt         ← thread-safe array-of-poses,
│           │   │                            UDP serialization
│           │   ├── TagConfig.kt          ← global config singleton
│           │   └── TagPose.kt            ← (tagId, 4×4 transform, ts)
│           ├── network/
│           │   └── UdpBroadcaster.kt     ← MulticastSocket sender
│           ├── service/
│           │   └── DetectionService.kt   ← see arch diagram above
│           └── ui/
│               ├── AppRoot.kt            ← root composable, screen routing
│               ├── AxisOverlayView.kt    ← classic Android View, Canvas-drawn
│               │                            axes + labels + field frame "F"
│               ├── LocalDetectionService ← CompositionLocal for the bound svc
│               ├── TagColors.kt          ← golden-angle hue index → ARGB
│               ├── camera/
│               │   └── CameraScreen.kt   ← bottom-sheet camera UI
│               ├── dashboard/
│               │   ├── DashboardScreen.kt    ← device list + FABs + sheets
│               │   ├── DeviceCard.kt         ← one card; battery, chips
│               │   ├── DeviceActionSheet.kt  ← per-device modal
│               │   ├── DevicesViewModel.kt   ← state hub for the dashboard
│               │   └── ModeSelectorPanel.kt  ← top-of-list mode + Apply
│               ├── settings/
│               │   └── SettingsScreen.kt ← read-only TagConfig display
│               └── theme/
│                   ├── Color.kt          ← palette + status colors
│                   └── Theme.kt          ← AtomTagTheme (forced dark M3)
├── build.gradle.kts                      ← root project (just plugin versions)
├── gradle.properties                     ← JVM args (4 GB heap, 1 GB metaspace)
└── settings.gradle.kts
```

---

## Coordinate frames (important domain knowledge)

There are **three** coordinate frames the code cares about:

1. **Camera frame** — output of `solvePnP` on each detected tag. X right, Y up,
   Z out of the page (away from the tag surface). All raw `TagPose.transform`
   values from `AprilTagDetector.detect` are in this frame.

2. **Zero Tag Frame** — origin tag's local coordinate frame. Computed by
   `PoseTransformer.transformToOriginFrame`: invert the origin tag's transform,
   then left-multiply each detected pose. The origin tag itself becomes
   identity in this frame.

3. **Field Frame** — the production frame everything else is reported in.
   Defined as a pure translation from the Zero Tag Frame by
   `(FIELD_FRAME_X_M, Y_M, Z_M)` (loaded from `tag_config.json` →
   `field_frame.{x_mm,y_mm,z_mm}`). For the soccer use case, this puts the
   origin in the middle of the field rather than at the goal-mounted Net tag.

   `PoseTransformer.transformToFieldFrame(detections)`:
   1. First runs `transformToOriginFrame`.
   2. Then subtracts the field offset from each pose's translation
      (rotation unchanged).
   3. Falls back to camera frame if the origin tag is not visible.

The visual overlay draws **per-tag** axes from each tag's local frame and a
**single field-frame "F"** triad rooted at the offset point (only when the
origin tag is detected — see `AprilTagDetector` populating
`DetectionResult.fieldFrameAxes` only for `tagId == ORIGIN_TAG_ID`).

### Origin pose smoothing

Even when the origin tag is physically still, `solvePnP` jitters frame to
frame. Translation jitter is small; rotation jitter, when extrapolated by the
field-frame offset (~450 mm), produces visible mm-level wobble on the field
axes (`offset · sin θ`). To fix this, `AprilTagDetector` runs an EMA on the
origin tag's `rvec`/`tvec` only — bots pass through unfiltered so robot motion
stays responsive.

```kotlin
ORIGIN_FILTER_ALPHA_R = 0.03   // rotation: heavy smoothing (~33-frame TC)
ORIGIN_FILTER_ALPHA_T = 0.15   // translation: moderate smoothing (~7-frame TC)
```

Lower alpha = smoother but slower to track if the camera or origin tag moves.
Bots use no filter.

---

## Detection pipeline

`DetectionService.analyzeFrame(imageProxy)` (called from a single-thread
`ImageAnalysis` analyzer):

1. **Convert** ImageProxy YUV → grayscale `Mat` (handles row stride).
2. **Initialize intrinsics** (once, using image width as focal length proxy).
3. **`AprilTagDetector.detect`**:
   - Full-frame `ArucoDetector.detectMarkers` (DICT_APRILTAG_36h11).
   - **ROI re-detection**: for each tag tracked last frame that wasn't found
     this frame, search a 2× expanded ROI around the previous bounding box.
     Tags missing >10 frames are dropped from tracking.
   - For each detected tag: `solvePnP` → rvec/tvec → 4×4 transform.
   - **EMA smooth** rvec/tvec for origin tag only (see above).
   - Project axis tips (origin, x, y, z) for visualization.
   - Project bottom-center for label placement.
   - **For origin tag**, also project field-frame axes
     (origin + offset → tip vectors).
4. **`PoseTransformer.transformToFieldFrame`** maps camera-frame poses into
   field-frame poses.
5. **Update `PoseVector`** with the transformed poses.
6. **Broadcast** via `UdpBroadcaster` if `BROADCAST_INTERVAL_MS` has elapsed.
7. **Update `CameraStats` StateFlow** (FPS, tag count, broadcast Hz, pose
   lines). Also bump the persistent notification with the current tag count.
8. **Update `AxisOverlayView`** (off the analysis thread, via `view.post`)
   with per-tag overlay data + field-frame axes.

### CameraX use cases

The service binds a `Preview` + `ImageAnalysis` to its own lifecycle. The
`Preview`'s `SurfaceProvider` is **set/cleared dynamically** from
`CameraScreen` so the camera keeps running when the modal is closed (no
preview to render, but analysis continues).

---

## UI layer

### Compose root

`AppRoot` switches between Dashboard and Settings via `mutableStateOf`. The
camera modal is independent — a `ModalBottomSheet` controlled by
`cameraOpen` state.

The detection service is exposed to the UI via `LocalDetectionService`
(a `CompositionLocal<DetectionService?>`). The provider is wired in
`MainActivity.setContent` and updated when the service connects/disconnects
via `ServiceConnection`.

### Dashboard (`DashboardScreen`)

- `DevicesViewModel` exposes:
  - `state: StateFlow<DashboardUiState>` — devices + a 500 ms `nowMs` ticker
    so age strings ("3s ago") update without server-side change.
  - `selectedMode: StateFlow<AppMode>`
  - `applyState: StateFlow<ApplyState>` — `inProgress`, `pendingMode`,
    `ackedCount`, `totalDevices`, `timedOut`. Selecting a different mode in
    the dropdown clears this back to default; selecting the same mode is a
    no-op.
  - `pingResults: Map<Int, PingResult>` — keyed by tagId
  - `restartStates: Map<Int, RestartState>` — Pending / Acknowledged /
    TimedOut. Cleared on sheet reopen via `clearCompletedRestart` so the
    button doesn't keep counting "Restarted Xs ago" forever.

- The Net (origin tag) is filtered out of the device list in
  `MockDeviceStateRepository.buildInitialDevices` (`it != ORIGIN_TAG_ID`).

- **`ModeSelectorPanel`** sits above the device cards as the first
  `LazyColumn` item. Apply button has four visual states: idle (Send icon +
  "Apply"), in-progress (spinner + ack count), success (green ✓ + count),
  timed-out (red ⚠ + count).

- **FAB row**: settings (bottom-left) and camera (bottom-right), both
  overlaid in a `Box` (Scaffold's single-FAB slot is bypassed). The list's
  bottom content padding is 96 dp so the last card isn't hidden behind the
  FABs.

### `DeviceCard`

- Color swatch on the left (4×54 dp).
- Name + tag ID column.
- Right column (with `padding(end = 54.dp)` to push left from the right
  edge): `StatusChip` (online/stale/offline + age) + `ModeChip` (mode label
  + health-colored dot).
- **Battery icon** is positioned absolute in the card's `Box`'s `TopEnd`
  corner, 16 dp inset. 28×14 dp Canvas: rounded rectangle border + tip,
  continuous fill from left, color interpolation red→yellow→green clamped
  at <20% / >80%.

### Battery curve

S-shaped cubic in normalized `t = (V - 6.1) / 2.2`:

```
f(t) = -1.43 t³ + 2.03 t² + 0.4 t
```

Anchors: f(6.1V) = 0, f(7.4V) = 0.65 (nominal sits above half), f(8.3V) = 1,
f'(0) = 0.4 (slow climb out of empty matches LiPo's flat ends). Constants
live in `DeviceCard.kt` (`VOLTAGE_FULL`, `VOLTAGE_EMPTY`, the cubic
coefficients).

### `DeviceActionSheet`

Modal bottom sheet that opens on card tap. Layout:
- Header: color swatch + name (with **pencil icon** to flip into a
  `OutlinedTextField` for renaming) + tag ID + last ping result; gear icon
  on the far right toggles the settings section.
- Action buttons: Ping, Restart. Restart button mutates based on
  `RestartState` (idle / pending spinner / Acknowledged ✓ / TimedOut ⚠).
- Status section: health-colored dot + "All good" / "Warning" / "Error",
  followed by bullet list of `device.statusMessages`.
- Settings section (when gear is on): color picker (8 presets + a sweep-
  gradient circle that swaps the section in-place for a 16-color extended
  palette + Cancel/OK), team selector chips.

**Critical**: do not stack a `Dialog` (or `AlertDialog`) on top of the
`ModalBottomSheet`. Nested popups have caused emulator-level segfaults in
software-rendering mode. The custom color picker is implemented as inline
content within the same sheet (not a Dialog) for this reason.

### Camera modal (`CameraScreen`)

Bottom sheet with `skipPartiallyExpanded = true` so it opens fully
expanded. Inside:
- `AndroidView { previewView }` and `AndroidView { overlayView }` stacked
  in a weight-1 `Box`.
- `DisposableEffect` handles attach/detach:
  ```kotlin
  service.attachPreview(previewView)
  service.attachOverlay(overlayView)
  onDispose {
      service.detachPreview()
      service.detachOverlay()
  }
  ```
- `ToggleChipRow`: three `FilterChip`s (Axes / Labels / Virtual bg).
- `StatsStrip`: one-line summary (`"30 fps · 3 tags · 20 Hz · ref:tag 0"`)
  collapsed by default; expands to per-line stats and the `poseLines` list.

### Theme

Forced dark Material 3. `AtomTagTheme` ignores the system light/dark
setting. Activity-level theme is `Theme.AppCompat.NoActionBar` (no system
action bar; Compose handles all chrome).

---

## Data layer

### `TagConfig` (singleton, loaded from `assets/tag_config.json`)

```json
{
  "origin_tag_id": 0,
  "multicast_group": "239.1.1.1",
  "multicast_port": 5000,
  "broadcast_interval_ms": 50,
  "field_frame": {"x_mm": 451.872, "y_mm": -118.1, "z_mm": -29.7},
  "tags": [
    {"id": 0, "size_meters": 0.08, "label": "Net"},
    {"id": 1, "size_meters": 0.05, "label": "Atom_1"},
    ...
  ]
}
```

- `MULTICAST_GROUP`, `MULTICAST_PORT`, `BROADCAST_INTERVAL_MS`
  control UDP output.
- `ORIGIN_TAG_ID` defines the Zero Tag.
- `FIELD_FRAME_X_M / Y_M / Z_M` is the offset from origin tag to field
  origin (loaded from mm and converted).
- Per-tag `size_meters` is the AprilTag side length in meters; required
  by `solvePnP`.
- Per-tag `label` appears in the UI (overridable per-device via
  `UserPreferences`).

### `UserPreferences` (singleton, JSON file)

`<filesDir>/device_overrides.json` stores per-tag user overrides:

```json
{
  "1": {"name": "Goalie", "color_argb": -65536, "team": "Red"},
  "3": {"color_argb": -16711936}
}
```

API:
- `init(context)` — load on app start
- `getOverride(tagId): DeviceOverride?`
- `setName / setColor / setTeam` — write through immediately

`MockDeviceStateRepository.buildInitialDevices` applies overrides on top
of `TagConfig` defaults; setters in the repo also write to
`UserPreferences`.

### `DeviceState` and the repository

`DeviceState` is the UI's view of one device:

```kotlin
data class DeviceState(
    val tagId: Int,
    val name: String,
    val colorArgb: Int,
    val batteryVolts: Float?,
    val lastSeenMs: Long?,
    val pose: TagPose?,
    val mode: AppMode? = null,
    val health: DeviceHealth = DeviceHealth.Ok,
    val statusMessages: List<String> = emptyList(),
    val team: String? = null,
)
```

`DeviceStatus` is **derived** at render time from `lastSeenMs` via
`statusAt(now)`: Online (<2 s), Stale (<10 s), Offline (≥10 s).

`DeviceStateRepository` is an interface; the only current impl is
`MockDeviceStateRepository`, which:
- Builds the device list from `TagConfig.allTagIds()` minus the origin.
- Ticks every 1 s: drifts battery, refreshes `lastSeenMs`.
- `applyMode`: simulates ~85% per-device ack within 200-1500 ms; the
  remaining 15% never ack (UI's 8 s timeout catches them).
- `setName / setColor / setTeam`: writes to `UserPreferences` + emits
  updated state.
- Mock-seeded state for demo: tag 2 is permanently Warning with two
  messages, tag 4 is Error with two messages.

When a real implementation arrives, swap in via the (currently default)
`DevicesViewModel` constructor parameter.

---

## Network (UDP)

`UdpBroadcaster`:
- `start()` opens a `MulticastSocket`.
- `broadcast(poseVector)` sends `poseVector.toBytes()` to
  `MULTICAST_GROUP:MULTICAST_PORT`.
- `stop()` closes.

`PoseVector.toBytes()` packet format (little-endian):

```
[numTags: int32][present:byte tagId:int32 transform: 16×float32] × numTags
```

`numTags` is fixed at `TagConfig.NUM_TAGS`. Slots without a current
detection have `present = 0` and the transform is zeroed.

The "broadcast Hz" stat in the camera modal is the actual outgoing
packet rate over a rolling 1-second window — not a simulated number.

---

## Service lifecycle

`DetectionService extends LifecycleService`:

- Started + bound from `MainActivity.onCreate` (only after camera
  permission granted).
- `onCreate`: starts foreground notification (channel `atomtag_detection`,
  importance LOW), starts UDP broadcaster, binds CameraX to its own
  lifecycle.
- `onStartCommand`: handles `ACTION_STOP` from the notification's Stop
  action.
- `onTaskRemoved`: stops self when the user swipes the app off the
  recents screen (so the service doesn't outlive the user's session).
- `onDestroy`: unbind camera, stop broadcaster, release detector,
  shutdown executor.
- `attachPreview / detachPreview / attachOverlay / detachOverlay`: bound
  by the camera modal's `DisposableEffect`. Detection keeps running with
  no preview when the modal is closed.

`MainActivity.onDestroy` unbinds the connection but does NOT stop the
service.

---

## Build variants

`app/build.gradle.kts` defines two variants:

| Variant | `USE_MOCK_DATA` | `SHOW_DEBUG_CONTROLS` |
|---------|-----------------|-----------------------|
| `debug` | true            | true                  |
| `release` | false         | false                 |

`MockDeviceStateRepository` is used unconditionally today; the build flag
is exposed for when a real implementation arrives. Switch variants in
Android Studio's Build Variants panel or via `./gradlew installRelease`
(release needs a signing config; debug-fallback is fine for local).

---

## Conventions and gotchas

### Dependencies

We deliberately **do not** depend on:
- `com.google.android.material:material` — only `Theme.AppCompat.*` from
  `androidx.appcompat` is needed.
- `androidx.viewpager2`, `androidx.fragment` — fully Compose now, no
  fragments or pagers.

`androidx.lifecycle:lifecycle-service` is required for `LifecycleService`;
adding it isn't optional.

### Compose root vs. fragments

There are no fragments. `MainActivity` is the only Activity. UI lives in
`AppRoot`. Don't reintroduce `Fragment` or `FragmentManager` — past
attempts to mix them with the camera modal broke the lifecycle.

### Camera frame access from analysis thread

`AxisOverlayView` is a classic Android `View`. The analysis callback runs
on the analysis executor (background). Always update the view via
`view.post { view.update(...) }` — never call `update()` directly from the
analysis thread.

### Origin pose filter is in `AprilTagDetector`

If you need to add per-tag filtering for bots, do it in a separate filter
keyed by tagId — do NOT extend the existing origin filter, which is
specifically designed to only smooth the origin.

### Modal bottom sheet + Dialog stacking

Already mentioned, repeating because it cost an emulator: do not show a
`Dialog` (or `AlertDialog`) while a `ModalBottomSheet` is on screen. Use
in-place content swapping inside the sheet instead.

### Themed icons / launcher

The launcher icon's adaptive XML keeps a `<monochrome>` layer, so apps
with the user's "Themed icons" setting on get a wallpaper-tinted version.
Do not assume the displayed icon matches the foreground PNG; on the
Pixel launcher, the bottom-right dock slot is the **predicted-app slot**
and gets a tinted ring regardless of the icon's contents.

### LiPo battery curve

The curve is a cubic anchored at f(6.1V)=0, f(7.4V)=0.65, f(8.3V)=1 with
a flat-end shape (S-curve). Tune voltages or recalculate the cubic if
the battery chemistry changes. Constants are at the bottom of
`DeviceCard.kt`.

### Gradle JVM args

`gradle.properties` sets `-Xmx4g -XX:MaxMetaspaceSize=1g`. The default
2 GB heap OOMs during APK packaging once the Compose dependency tree is
in place. Don't shrink this.

---

## What's stubbed (future work)

- **Real `DeviceStateRepository` impl** — currently only `Mock` exists.
  The real one will receive battery voltages, modes, health, and status
  messages over UDP from the actual robots and update its StateFlow
  accordingly. The mock's interface is the contract.
- **Real ping/restart over the wire** — `DevicesViewModel.ping/restart`
  simulate stochastic timing. Real implementation should send UDP
  control packets and parse acks.
- **Real "apply mode" wire format** — `applyMode` simulates per-device
  ack timing. Define a UDP control packet format and have devices
  acknowledge by reporting their current mode via the regular telemetry
  stream.
- **Mode-specific settings UI** — the mode dropdown exists; per-mode
  tunable controls below it are not yet implemented. The framework is
  the `ModeSelectorPanel` composable; add a section underneath the
  dropdown that switches on `selectedMode`.
- **Virtual background overlay** — toggle in the camera modal exists
  (`drawVirtualBackground`). The renderer doesn't draw anything for it
  yet.
- **Settings screen edits** — `SettingsScreen` displays `TagConfig`
  values read-only. Editing requires writing back to disk and reloading.
- **Camera intrinsics calibration** — currently uses image width as
  focal length proxy. A real calibration (checkerboard, OpenCV
  `calibrateCamera`) would improve pose accuracy, especially in meters.

---

## Where to look first

- **App won't start / black screen** → `MainActivity.kt` perm flow,
  service binding, `OpenCVLoader.initLocal()`.
- **Camera crashes on open** → emulator camera config (must be
  `emulated` or `webcam0`), API level (avoid Android 17 preview images;
  use API 35), CameraX version (1.4.x for newer dynamic-range profile
  IDs).
- **Field axes jumping around** → `AprilTagDetector` filter alphas
  (`ORIGIN_FILTER_ALPHA_R / _T`).
- **Bot pose values wrong** → `PoseTransformer.transformToFieldFrame`,
  `TagConfig.FIELD_FRAME_*_M`, `tag_config.json` `field_frame` block.
- **Battery icon weird** → `DeviceCard.batteryFill` cubic + voltage
  constants.
- **Dashboard missing a device / showing "Net"** →
  `MockDeviceStateRepository.buildInitialDevices` filter,
  `TagConfig.ORIGIN_TAG_ID`.
- **User-edited names/colors lost on restart** →
  `UserPreferences.init(context)` not called, or `device_overrides.json`
  not writable in `filesDir`.
- **UDP broadcast not arriving** → check `MULTICAST_GROUP / PORT`,
  network has multicast enabled, listener has joined the group.
