import asyncio
import websockets
import json
import numpy as np

async def send_fake_ecg():
    uri = "ws://localhost:8000/ws/ecg/"
    try:
        async with websockets.connect(uri) as websocket:
            phase = 0
            while True:
                t = np.linspace(0, 1, 250)
                ecg = np.sin(2 * np.pi * 1.7 * t + phase) + 0.05 * np.random.randn(250)
                phase += 0.2

                print("📤 ECG out:", ecg[:5])
                await websocket.send(json.dumps({
                    "ecg": ecg.tolist()
                }))

                await asyncio.sleep(0.5)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(send_fake_ecg())