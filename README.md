# ECG Holter Monitor

A wearable, single-lead ECG (electrocardiogram) monitor that streams a patient's
heart waveform over Wi-Fi to a remote dashboard in real time.

An **ESP32** with an **AD8232** ECG front-end is worn on the body, samples the
heart signal, and sends it to a **Django** server. The server cleans the signal
with a digital filter and broadcasts it to any browser, which draws a live
scrolling waveform.

```
Patient + AD8232  ──►  ESP32 (Wi-Fi)  ──►  Django + Channels  ──►  Browser (Chart.js)
   electrodes          250 Hz, JSON         8-stage DSP filter      live waveform
```

## How it works

1. The ESP32 reads the analog ECG on **GPIO36** at ~250 Hz, in batches of 250
   samples (~1 second each).
2. It sends each batch as JSON `{"ecg": [...]}` over a WebSocket to the server.
3. The Django `ECGConsumer` filters the batch (baseline/powerline/wavelet
   denoising, R-peak preservation) and broadcasts `{"filtered": [...]}` to all
   connected dashboards.
4. The browser keeps a rolling buffer and redraws the trace live.

## Hardware & wiring

| Part            | Detail                                   |
|-----------------|------------------------------------------|
| Microcontroller | ESP32 (Wi-Fi 2.4 GHz)                    |
| ECG front-end   | AD8232 single-lead module                |
| ADC input       | GPIO36 (ADC1_CH0 / "VP"), 12-bit, 0–4095 |
| Electrodes      | 3 snap electrodes (RA, LA, RL reference) |

Only **three wires** are needed between the AD8232 and the ESP32 (the firmware
only reads the analog output):

| AD8232  | →  | ESP32  |
|---------|----|--------|
| OUTPUT  | →  | GPIO36 |
| 3.3V    | →  | 3V3    |
| GND     | →  | GND    |

The AD8232 `LO+`, `LO-`, and `SDN` (lead-off / shutdown) pins are left
unconnected. Attach the three electrodes to the body (RA = right, LA = left,
RL = right-leg reference).

## Project layout

```
HolterMonitor/
├── arduino/websocket_ecg_send_data/   # ESP32 firmware (.ino)
├── holter_monitor/                    # Django project
│   ├── ecg/                           # consumer, DSP filter, dashboard template
│   ├── send_fake_data.py              # hardware-free ECG simulator
│   └── manage.py
└── doc/                               # full LaTeX documentation
```

## Run it

### 1. Start the server

```bash
cd holter_monitor
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install django channels daphne numpy scipy PyWavelets websockets
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Open **http://localhost:8000/** (or `http://<server-ip>:8000/` from another
device) to view the dashboard.

### 2a. With the ESP32 (real ECG)

1. Open `arduino/websocket_ecg_send_data/websocket_ecg_send_data.ino` in the
   Arduino IDE (with the ESP32 board package + `ArduinoWebsockets` and
   `ArduinoJson` libraries).
2. Set your Wi-Fi `ssid`, `password`, and `websocket_server_host` (the server's
   LAN IP) at the top of the sketch.
3. Wire the AD8232 as above, then upload to the ESP32.
4. The dashboard waveform should start scrolling.

### 2b. Without hardware (simulator)

With the server running, in another terminal:

```bash
cd holter_monitor
python send_fake_data.py
```

This streams a synthetic ECG-like signal to the dashboard so you can test the
whole pipeline with no electronics.

## Documentation

Full technical documentation (architecture, signal-processing pipeline, wiring
schematic, build/run details) lives in [`doc/`](doc/) as LaTeX. Build it with
`latexmk -pdf main.tex`, or upload the folder to [Overleaf](https://overleaf.com).

## Tech stack

ESP32 (Arduino C++) · Django 5.1 + Channels + Daphne (ASGI/WebSockets) ·
NumPy / SciPy / PyWavelets (DSP) · Chart.js 4.4 (dashboard).

## Notes

This is a proof of concept, not a medical device. The firmware credentials and
Django settings (`DEBUG=True`, open hosts) are for development only — add
authentication, TLS, and per-patient data handling before any real use.
