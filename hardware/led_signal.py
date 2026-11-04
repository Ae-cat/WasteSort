
# Python code categorization will be sent to arduino code to signal LED light up
# Arduino code is written on the Arduino IDE desktop app
import serial
import time

# Test for import success
# print("PySerial Works!")

# ARDUINO CONNECTION
try:
    arduino = serial.Serial(
        '/dev/ttyACM0',
        9600,
        timeout=1
    )
    # Arduino resets when serial connection opens
    time.sleep(2)
    print("Successfully connected to WasteSort Arduino!")

except Exception as e:
    print("Could not connect to Arduino:", e)
    arduino = None

# LED CONTROL
def trigger_bin_light(classification_result):

    if arduino is None:
        return
    # Convert ML classification into Arduino signal
    if classification_result == "recyclable":
        arduino.write(b'R')
        print("Recyclable detected - GREEN LEDs")
    elif classification_result == "landfill":
        arduino.write(b'L')
        print("Landfill detected - BLUE LEDs")
    elif classification_result == "compost":
        arduino.write(b'C')
        print("Compost detected - WHITE LEDs")
    else:
        # Unknown classification: turn off LEDs
        arduino.write(b'X')
        print("Unknown classification - LEDs OFF")

# SIMULATION TEST
if __name__ == "__main__":
    print("Testing WasteSort LED system...")

    # Test recycling
    trigger_bin_light("recyclable")
    time.sleep(3)

    # Test landfill
    trigger_bin_light("landfill")
    time.sleep(3)

    # Test compost
    trigger_bin_light("compost")
    time.sleep(3)

    # Turn everything off
    arduino.write(b'X')
    print("LED test complete!")