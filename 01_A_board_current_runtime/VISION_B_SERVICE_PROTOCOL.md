# A/B vision service contract (version 1)

This A-board package owns ECG acquisition/inference, the LVGL UI, HDMI, TFT,
GT911 touch, and SW0 display selection. B owns the USB camera, image capture,
pose/electrode inference, and any visual accelerator. The A application never
opens `/dev/video0` and never loads a vision model.

## Connection

- A is the TCP client; B listens on the configured address and port (default
  `192.168.2.100:8765`). Set `PULSEATLAS_B_HOST` / `PULSEATLAS_B_PORT`, or edit
  `vision_peer.json` on A. The address is a placeholder until B is configured.
- A reconnects every 1.5 seconds after a dropped connection. A connection or B
  failure does not block ECG, HDMI, TFT, or touch.
- There is no login in this prototype. Keep the service on the trusted wired
  LAN; do not expose the port to an untrusted network.

## Packet framing

Every packet is `4-byte unsigned big-endian header length`, UTF-8 JSON header,
then exactly `payload_len` raw payload bytes. Header limit is 64 KiB and payload
limit is 2 MiB. Unknown message types close the connection. B should send a
status/heartbeat at least once every 4 seconds. The A client treats a packet
that cannot complete in 5 seconds as a disconnected service.

On connect A sends a passive subscription while the application is on its
home page; this keeps status updates live without opening the camera or model:

```json
{"type":"subscribe","version":1,"preview":"jpeg","active":false,"payload_len":0}
```

A sends `{"type":"command","action":"start"}` when the Vision page is
opened and `stop` when its start/stop control is toggled off. `payload_len` is
added automatically. The B R2 status packet checks the configured
`/dev/videoN` device node on each report, so `camera_connected` stays live while
capture and inference are idle; the node check does not open the camera.

B may send a `hello`, `ack`, or `pong` with no payload, plus status and preview
packets. A `status` packet has no payload and uses these fields:

```json
{
  "type":"status", "payload_len":0,
  "camera_connected":true, "camera_frames":120, "camera_fps":24.0,
  "inference_fps":4.2, "inference_ms":238.0,
  "inference_running":false, "inference_count":20,
  "result_age_ms":120, "result_status":"人体与贴片已定位",
  "confidence":0.91, "stale":false, "detected":true,
  "marker_count":10, "pose_valid":true, "status":"视觉服务在线",
  "error":null,
  "connector_status":"电极片检查完成", "connector_error":null,
  "connector_inference_ms":18.5, "connector_frame_age_ms":42.0,
  "connector_correct":8, "connector_adjust":1, "connector_unknown":1,
  "connector_per_lead":{
    "RA":{"status":"correct","confidence":0.94},
    "LA":{"status":"adjust","confidence":0.83},
    "RL":{"status":"unknown","confidence":0.0}
  }
}
```

`camera_fps` means frames captured on B; `inference_fps` means completed model
inferences, not preview refresh. `result_age_ms` is age of the observation that
produced the displayed landmarks. `marker_count` counts valid projected lead
markers; `pose_valid` means shoulder/hip geometry passed B's quality checks.
When the camera or model is unavailable, B must send a status with
`camera_connected:false` or an explanatory `error`, not a fabricated result.

Electrode presence is reported per calibrated point using `correct`, `adjust`,
or `unknown`. `connector_correct`, `connector_adjust`, and
`connector_unknown` summarize the ten expected sites (RA, LA, RL, LL, V1–V6).
The A UI renders all ten labels on HDMI and a count plus the specific review and
pending labels on TFT. These fields report visual pad/color matches near the
calibrated points; they do not establish skin contact, electrical continuity,
or clinical correctness. A must treat absent per-lead results or stale results
as pending rather than reusing them as a new detection.

A preview packet has `type:"frame"`, a monotonically increasing `seq`, the
JPEG byte count in `payload_len`, and a JPEG payload. Send a downscaled preview
(recommended max 640×480, 5–10 FPS); inference can run at a different cadence.
The client retains only the latest decoded preview frame, not a queue.

## Calibration handoff

`vision_guide/calibration/torso_lead_calibration.json` and
`chest_lead_calibration.json` are the restored manual version-2 calibration
results. They are retained here as integration references. B should consume
these files with the same shoulder/hip torso-coordinate convention and V3
midpoint rule, then report only valid, fresh observations. The A app renders
what B reports; it does not reproject or reinterpret coordinates.

## Offline behavior

Until B is implemented and running, A's Vision page displays “等待 B 板视觉服
务” and the configured endpoint, with no camera or model loaded. This is the
expected A-only state, not an ECG fault. Existing ECG, TFT, touch, HDMI, and
SW0 functions do not depend on the vision service.
