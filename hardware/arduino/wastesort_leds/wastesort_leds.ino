// Arduino library required: Adafruit NeoPixel by Adafruit
#include <Adafruit_NeoPixel.h>

// LED CONFIGURATION
#define LED_PIN 6
// Total LEDs used across the three compartments
#define NUM_PIXELS 36
// Number of LEDs assigned to each compartment
#define LEDS_PER_BIN 12

// LED strip configuration
Adafruit_NeoPixel strip(
    NUM_PIXELS,
    LED_PIN,
    NEO_GRB + NEO_KHZ800
);

// SETUP
void setup() {
    // Raspberry Pi communicates through USB serial
    Serial.begin(9600);

    // Initialize LED strip
    strip.begin();

    // Start with all LEDs turned off
    strip.clear();
    strip.show();

    Serial.println("WasteSort LED system ready!");
}

// LED FUNCTIONS
// Turn off every LED
void turnOffAll() {
    strip.clear();
    strip.show();
}

// Light up one specific compartment
void lightBin(
    int startPixel,
    int endPixel,
    uint32_t color
) {
    // Turn off all other compartments first
    strip.clear();

    // Light only the selected compartment
    for (int i = startPixel; i < endPixel; i++) {
        strip.setPixelColor(i, color);
    }
    // Update physical LED strip
    strip.show();
}

// MAIN LOOP
void loop() {
    if (Serial.available() > 0) {
        // Read classification sent by Raspberry Pi
        char signal = Serial.read();
        // RECYCLING: First compartment - GREEN
        if (signal == 'R') {
            lightBin(
                0,
                LEDS_PER_BIN,
                strip.Color(0, 255, 0)
            );
            Serial.println("Recycling detected!");
        }
        // LANDFILL: Second compartment - BLUE
        else if (signal == 'L') {
            lightBin(
                LEDS_PER_BIN,
                LEDS_PER_BIN * 2,
                strip.Color(0, 0, 255)
            );
            Serial.println("Landfill detected!");
        }
        // COMPOST: Third compartment - WHITE
        else if (signal == 'C') {
            lightBin(
                LEDS_PER_BIN * 2,
                NUM_PIXELS,
                strip.Color(255, 255, 255)
            );
            Serial.println("Compost detected!");
        }
        // TURN OFF ALL LIGHTS
        else if (signal == 'X') {
            turnOffAll();
            Serial.println("All LEDs off!");
        }
    }
}