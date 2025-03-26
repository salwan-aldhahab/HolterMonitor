# from channels.generic.websocket import AsyncWebsocketConsumer
# import json
# from .filters import filter_ecg_signal

# class ECGConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()

#     async def disconnect(self, close_code):
#         pass

#     async def receive(self, text_data):
#         data = json.loads(text_data)
#         raw = data.get("ecg", [])

#         filtered = filter_ecg_signal(raw)

#         await self.send(text_data=json.dumps({
#             "filtered": filtered
#         }))

# from channels.generic.websocket import AsyncWebsocketConsumer
# import json
# from .filters import filter_ecg_signal

# class ECGConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         print("🚀 WebSocket Connected")
#         await self.accept()

#     async def disconnect(self, close_code):
#         print(f"❌ WebSocket Disconnected: {close_code}")

    # async def receive(self, text_data):
    #     try:
    #         print("📥 Received data")
    #         data = json.loads(text_data)
    #         raw = data.get("ecg", [])
    #         print(f"➡️ Raw length: {len(raw)}")

    #         filtered = filter_ecg_signal(raw)
    #         print("✅ Filtered ECG computed")

    #         await self.send(text_data=json.dumps({
    #             "filtered": filtered
    #         }))
    #         print("📤 Sent filtered ECG")

    #     except Exception as e:
    #         print(f"💥 Error in WebSocket receive: {e}")
    
    # async def receive(self, text_data):
    #     try:
    #         print("📥 Received WebSocket:", text_data)
    #         data = json.loads(text_data)
    #         raw = data.get("ecg", [])

    #         # Optional: print preview
    #         print(f"➡️ ECG sample (first 5): {raw[:5]}")

    #         # Optionally skip filtering for now:
    #         filtered = raw
    #         await self.send(text_data=json.dumps({
    #             "filtered": filtered
    #         }))
    #     except Exception as e:
    #         print(f"💥 Error: {e}")
    
    # async def receive(self, text_data):
    #     try:
    #         print("📥 Received WebSocket:", text_data)
    #         data = json.loads(text_data)
    #         raw = data.get("ecg", [])

    #         print(f"➡️ ECG sample (first 5): {raw[:5]}")

    #         # Disable filtering for now
    #         filtered = raw

    #         await self.send(text_data=json.dumps({
    #             "filtered": filtered
    #         }))
    #         print("📤 Sent to frontend:", filtered[:5])

    #     except Exception as e:
    #         print(f"💥 Error: {e}")
    
    # async def receive(self, text_data):
    #     try:
    #         print("📥 Received WebSocket:", text_data)
    #         data = json.loads(text_data)
    #         raw = data.get("ecg", [])

    #         print("➡️ First 5 samples:", raw[:5])

    #         await self.send(text_data=json.dumps({
    #             "filtered": raw  # Send unfiltered for now
    #         }))
    #         print("📤 Sent to frontend:", raw[:5])
    #     except Exception as e:
    #         print(f"💥 WebSocket error: {e}")
    
from channels.generic.websocket import AsyncWebsocketConsumer
import json

from .filters import filter_ecg_signal

class ECGConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("ecg_group", self.channel_name)
        await self.accept()
        print("🚀 WebSocket Connected")

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("ecg_group", self.channel_name)
        print("❌ WebSocket Disconnected:", close_code)

    # async def receive(self, text_data):
    #     print("📥 Received WebSocket:", text_data)
    #     try:
    #         data = json.loads(text_data)
    #         ecg_data = data.get("ecg", [])
    #         print("➡️ First 5 samples:", ecg_data[:5])

    #         # Broadcast to group (both frontend & backend can receive)
    #         await self.channel_layer.group_send(
    #             "ecg_group",
    #             {
    #                 "type": "send_ecg",
    #                 "data": ecg_data
    #             }
    #         )
    #     except Exception as e:
    #         print(f"💥 WebSocket Error: {e}")
    
    async def receive(self, text_data):
        print("📥 Received WebSocket:", text_data)
        try:
            data = json.loads(text_data)
            raw = data.get("ecg", [])

            print("➡️ Raw length:", len(raw))

            # ✅ Apply your filter
            filtered = filter_ecg_signal(raw)
            print("✅ Filtered ECG computed")

            # Broadcast to group (frontend)
            await self.channel_layer.group_send(
                "ecg_group",
                {
                    "type": "send_ecg",
                    "data": filtered
                }
            )

        except Exception as e:
            print(f"💥 WebSocket Error: {e}")


    async def send_ecg(self, event):
        filtered = event["data"]
        await self.send(text_data=json.dumps({"filtered": filtered}))
        print("📤 Broadcasted filtered ECG")
