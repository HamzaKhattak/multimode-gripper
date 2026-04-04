#include <FastLED.h>

#define LED_PIN     12
#define NUM_LEDS    12
CRGB leds[NUM_LEDS];
const char DEVICE_CODE[] = "RGBBL-001";

enum LedMode {
  MODE_OFF,
  MODE_WHITE,
  MODE_RAINBOW,
  MODE_RGB_SPLIT,
};

LedMode currentMode = MODE_WHITE;
LedMode lastAppliedMode = MODE_OFF;

void setLeds(bool on) {
  CRGB color = on ? CRGB(255, 255, 255) : CRGB::Black;
  fill_solid(leds, NUM_LEDS, color);
}

void setRgbSplit() {
  int segmentSize = NUM_LEDS / 3;
  for (int i = 0; i < NUM_LEDS; i++) {
    if (i < segmentSize) {
      leds[i] = CRGB::Red;
    } else if (i < segmentSize * 2) {
      leds[i] = CRGB::Green;
    } else {
      leds[i] = CRGB::Blue;
    }
  }
}

void setRainbow() {
  fill_rainbow(leds, NUM_LEDS, 0, 255 / NUM_LEDS);
}

void applyCurrentMode() {
  if (currentMode == lastAppliedMode) {
    return;
  }

  switch (currentMode) {
    case MODE_OFF:
      setLeds(false);
      break;
    case MODE_WHITE:
      setLeds(true);
      break;
    case MODE_RAINBOW:
      setRainbow();
      break;
    case MODE_RGB_SPLIT:
      setRgbSplit();
      break;
  }

  lastAppliedMode = currentMode;
}

bool parseSerialCommand() {
  bool commandAccepted = false;

  while (Serial.available() > 0) {
    char command = Serial.read();

    if (command == '\n' || command == '\r') {
      continue;
    }

    if (command == '1') {
      currentMode = MODE_WHITE;
      Serial.println("LEDs: WHITE");
      commandAccepted = true;
    } else if (command == '0') {
      currentMode = MODE_OFF;
      Serial.println("LEDs: OFF");
      commandAccepted = true;
    } else if (command == 'r' || command == 'R') {
      currentMode = MODE_RAINBOW;
      Serial.println("LEDs: RAINBOW");
      commandAccepted = true;
    } else if (command == 's' || command == 'S') {
      currentMode = MODE_RGB_SPLIT;
      Serial.println("LEDs: RGB SPLIT");
      commandAccepted = true;
    }
  }

  return commandAccepted;
}

void waitForInitialCommand() {
  const unsigned long announceIntervalMs = 1000;
  unsigned long lastAnnounceMs = 0;

  while (true) {
    unsigned long nowMs = millis();
    if (nowMs - lastAnnounceMs >= announceIntervalMs) {
      Serial.println(DEVICE_CODE);
      lastAnnounceMs = nowMs;
    }

    if (parseSerialCommand()) {
      return;
    }
  }
}

void setup() {
  Serial.begin(115200);
  FastLED.addLeds<WS2812, LED_PIN, GRB>(leds, NUM_LEDS);
  currentMode = MODE_OFF;
  applyCurrentMode();
  FastLED.show();
  waitForInitialCommand();
  applyCurrentMode();
}

void loop() {
  parseSerialCommand();
  applyCurrentMode();
  FastLED.show();
}
