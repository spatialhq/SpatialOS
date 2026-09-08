# Capturing raw data

`spatial-os` is a backend — it processes scans, it doesn't capture them. To
produce a raw scan on an iPhone with a LiDAR sensor (iPhone 12 Pro or newer,
or an iPad Pro with LiDAR):

1. Install the **3D Scanner App** and follow the
   [iPhone LiDAR tutorial on protocols.io](https://www.protocols.io/view/iphone-lidar-tutorial-yxmvm21w9g3p/v1)
   to record a scan of a room or object.
2. Export the scan as **"All Data"** — this produces the per-frame RGB/depth/
   pose files that `spatial-os ingest --format 3dscannerapp` expects
   (`frame_NNNNN.jpg`, `depth_NNNNN.png`, `frame_NNNNN.json`, `info.json`).
3. Transfer the exported folder to the machine running `spatial-os` (AirDrop,
   Files app share, cable, etc.).

See [github.com/laanlabs/3dScannerApp_samples](https://github.com/laanlabs/3dScannerApp_samples)
for example exports and reference parsing code from the app's developer.

If your export's file names or JSON fields differ from what's documented in
`src/spatial_os/ingest/threed_scanner_app.py` (the app's export format has
changed across versions), adjust that adapter to match — it's a thin,
self-contained module.
