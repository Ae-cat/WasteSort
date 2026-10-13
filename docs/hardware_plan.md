# Hardware Plan

## Goal

Determine the hardware required for the WasteSort prototype.

## Planned Hardware

- Edge computer
    ### Raspberry Pi 5 (4GB)
    Reasons:
    - 4GB of RAM should be enough for our image classification system.
    - It can run Python, OpenCV, and a lightweight machine learning model locally.
    - Supports USB cameras for capturing waste images.
    - Has GPIO pins that can control LEDs and hardware.
    - Connects to a display showing the predicted waste item and disposal category.
    - Price: about $110.
    - More affordable than higher AI computers such as the NVIDIA Jetson.
    
    If the testing shows that AI inference is too slow, an AI accelerator (70$) it could be considered later.

    ### Other Options
    - Raspberry Pi 5 (2GB): it is cheaper, but provides less memory for running the model, camera processing, and user interface together. Could potentially not work.
    - Raspberry Pi 5 (8GB): provides more memory but currently costs approximately $175, so the additional cost may not be necessary for the prototype.
    - NVIDIA Jetson Orin Nano Super: significantly more powerful for AI workloads, but very expensive.

    https://www.pishop.us/product-category/raspberry-pi/raspberry-pi-5/raspberry-pi-5-boards 

- Camera
    ### Raspberry Pi Camera Module 3
    Reasons:
    - 12 MP camera with autofocus.
    - Capture high quality images for waste classification.
    - Designed specifically for Raspberry Pi.
    - Small size makes it easy to integrate into the final WasteSort prototype.
    - Connects directly to the Raspberry Pi using a camera ribbon cable.
    - Supports 1080p video up to 50 fps.
    - Price: about $25.
    - More compact and cheaper than traditional USB webcam.

    https://www.raspberrypi.com/products/camera-module-3

- Display
    ### Raspberry Pi Touch Display 2 (5-inch)
    Reasons:
    - 5-inch the screen is big enough to neatly display the item, and the disposal category.
    - 720 × 1280 resolution.
    - Designed for Raspberry Pi and compatible with the Raspberry Pi 5.
    - Compact enough to integrate into the final WasteSort prototype.
    - Touchscreen capability is not required for the MVP, but could be useful for future features such as a "Scan Again" button. 
    - Price: about $40.

    https://www.raspberrypi.com/products/touch-display-2

- LEDs

    ### Plusivo 5mm LED Assortment Kit
    WasteSort would use 3 LEDs to visually represent each disposal category.

    - Blue LED: Recyclable
    - Green LED: Compost
    - Red LED: Landfill
    - The LEDs are controlled through the Raspberry Pi GPIO pins.
    - 220-ohm resistors will be used to limit current and protect the LEDs.
    - Using separate LEDs makes each disposal category easy to identify in the prototype.
    - The Plusivo kit includes 5mm LEDs in multiple colors and 220-ohm resistors.
    - Current kit price: approximately $13.

    https://www.plusivo.com/electronics-kit/29-plusivo-5mm-led-assortment-kit-500pcs-with-bonus-pcb-and-220-resistors100pcs.html

- Servo motor
    ### TowerPro SG92R Micro Servos
    WasteSort would need three servo motors, one for each category. After the AI identifies the waste category, the corresponding servo will open the correct lid.

    - 3 TowerPro SG92R micro servos.
    - One servo for Recyclable, one for Compost, and one for Landfill.
    - About 180 degrees for rotation.
    - Small and lightweight.
    - Controlled by the Raspberry Pi.
    - Price: about $5.95 each, or $17.85 for three.
    - The prototype lids have to be lightweight so the micro servos can operate them well.
    - The servos will require an appropriate power setup rather than being powered directly from the Raspberry Pi GPIO pins for better result. 

    https://www.adafruit.com/product/169
    

- Power Supply
    ### Raspberry Pi Power
    The Raspberry Pi 5 will use the official Raspberry Pi 27W USB-C Power Supply.

    - Designed specifically for the Raspberry Pi 5.
    - Provides up to 5V/5A.
    - Powers the Raspberry Pi and connected components such as the camera and display.
    - price: about $12.

    https://www.raspberrypi.com/products/27w-power-supply 


    ### Servo Controller (Adafruit PCA9685)
    The Adafruit PCA9685 control the three servo motors.

    - Controls the three servo motors from the Raspberry Pi.
    - The three SG92R servos can connect directly to the servo headers on the board.
    - Communicates with the Raspberry Pi using I2C.
    - Keeps servo control organized and reduces the number of GPIO pins needed.
    - Allows the servos to use a separate power source instead of drawing motor power from the Raspberry Pi.
    - Price: about $14.95.

    https://www.adafruit.com/product/815 

    ### Servo Power
    The 3 SG92R servo motors have to use another 5V power supply to prevent the motors from drawing power directly from the Raspberry Pi.

    - Provides 5V power with up to 4A of current for the servo system.
    - Price: about $14.95.
    - A 2.1mm female DC barrel jack to screw-terminal adapter will connect the power supply to the PCA9685 servo controller.
    - Adapter price: about $2.00.
    - The PCA9685 will distribute power and control signals to the three SG92R servos.
    - This keeps the servo motor power separate from the Raspberry Pi power supply and reduces the risk of voltage drops or Raspberry Pi resets.

    https://www.adafruit.com/product/1466
    https://www.adafruit.com/product/368
    
    ## Raspberry Pi Cooling- Optional - Raspberry Pi Active Cooler
    The Raspberry Pi 5 can work without cooling, this is not required for the basic WasteSort prototype. Though active cooling is recommended because WasteSort will run camera processing and AI model inference for extended periods.

    Reasons:
    - Designed for the Raspberry Pi 5.
    - Helps prevent thermal throttling, where the Raspberry Pi reduces performance when it becomes too hot.
    - Helps maintain more consistent performance during extended testing and demonstrations.
    - Compact enough to integrate into the final WasteSort prototype.
    - Recommended for reliability during sustained AI and camera workloads, but not required for basic operation.
    - Current price: approximately $10.

    https://www.microcenter.com/product/671930/raspberry-pi-5-active-cooler


## Estimated Cost
    - Estimated hardware cost: $259.75


## Week 1 Deliverable

Hardware team will research the required components and create
a preliminary hardware plan.