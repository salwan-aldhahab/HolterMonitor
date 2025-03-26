#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <ArduinoJson.h>

using namespace websockets;

const char* ssid = "Sal-S23";
const char* password = "S6478987757A";

// WebSocket server URL (replace with your PC's IP!)
const char* websocket_server_host = "192.168.133.146";  // ← update to your PC's IP
const int websocket_server_port = 8000;

WebsocketsClient client;

#define ECG_PIN 36
#define SAMPLE_COUNT 250
#define SAMPLE_DELAY_MS 4  // ~250Hz sampling rate

void connectToWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected. IP: " + WiFi.localIP().toString());
}

bool connectToWebSocket() {
  String ws_url = "ws://" + String(websocket_server_host) + ":8000/ws/ecg/";
  return client.connect(ws_url);
}

void setup() {
  Serial.begin(115200);
  connectToWiFi();

  if (connectToWebSocket()) {
    Serial.println("✅ Connected to WebSocket server!");
  } else {
    Serial.println("❌ WebSocket connection failed.");
  }
}

void loop() {
  if (!client.available()) {
    Serial.println("🔌 Reconnecting WebSocket...");
    connectToWebSocket();
    delay(1000);
    return;
  }

  // Collect 250 ECG samples
  int ecg_values[SAMPLE_COUNT];
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    ecg_values[i] = analogRead(ECG_PIN);
    delay(SAMPLE_DELAY_MS);
  }

  // Build JSON
  StaticJsonDocument<4096> doc;
  JsonArray ecg_array = doc.createNestedArray("ecg");
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    ecg_array.add(ecg_values[i]);
  }

  String json;
  serializeJson(doc, json);

  // Send over WebSocket
  client.send(json);
  Serial.println("📤 ECG Sent: " + json.substring(0, 80) + "...");

  // Handle responses (optional)
  client.poll();
}