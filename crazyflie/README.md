# Crazyflie Real-Camera Pipeline

This directory is independent from the PyBullet simulation code.

## Structure

- `test_camera_stream.py`: verifies the XIAO ESP32S3 MJPEG stream.
- `test_real_perception.py`: runs color-obstacle detection on the real stream.
- `perception.py`: reusable color perception.
- `state_estimator.py`: camera-relative obstacle state estimation.
- `safety_oracle.py`: safety decision logic.
- `results/`: real experiment outputs.

## Install

```powershell
pip install -r requirements.txt
```

## Test the real camera stream

```powershell
python test_camera_stream.py
```

Explicit stream URL:

```powershell
python test_camera_stream.py --stream-url "http://192.168.214.170:81/stream"
```

## Test real-camera perception

```powershell
python test_real_perception.py
```

Use a different physical obstacle width:

```powershell
python test_real_perception.py --obstacle-diameter-m 0.20
```

Press `Q` to stop either program.
