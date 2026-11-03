# Raspberry Pi Setup Checklist for Ariella

This is a written handoff only.
No Pi setup, imaging, package installation, SSH connection, or camera test was executed during the laptop work.
Record completion and any errors on the actual hardware.

## Prepare the OS

Follow the official [Raspberry Pi setup documentation](https://www.raspberrypi.com/documentation/computers/getting-started.html).

- [ ] Gather the Pi 5, suitable power supply, microSD card and reader, camera, and network details.
- [ ] Install Raspberry Pi Imager on your laptop and select Raspberry Pi 5 and Raspberry Pi OS (64-bit).
Choose the desktop edition for local preview testing.
- [ ] Select the intended microSD card carefully; writing the image erases that card.
- [ ] In Imager customisation, set hostname `wastesort`, your own username and password, time zone, keyboard, Wi-Fi SSID/password, and regulatory country.
Do not store passwords in this repository.
- [ ] Enable SSH, preferably with your public key, and write/verify the image.
- [ ] Insert the card and boot, then test network reachability and SSH from the laptop using your configured username:

```sh
ssh YOUR_USERNAME@wastesort.local
```

If mDNS fails, use the Pi's IP address.
If campus Wi-Fi isolates devices, use an allowed hotspot or Ethernet network.

## Attach and test the camera

Follow the official [camera installation and software documentation](https://www.raspberrypi.com/documentation/accessories/camera-software.html).

- [ ] Shut down and disconnect power before attaching the camera.
Use the Pi 5 22-pin to camera 15-pin cable, seat it straight, and latch both connectors according to the official diagrams.
- [ ] Reconnect power, update the OS, and install the camera and Python tools on the Pi:

```sh
sudo apt update
sudo apt full-upgrade
sudo apt install rpicam-apps python3-picamera2 python3-venv python3-pip python3-opencv git
```

- [ ] With a display connected to the Pi, verify detection, preview, and capture:

```sh
rpicam-hello --list-cameras
rpicam-hello --timeout 5000
rpicam-still --nopreview --output ~/camera-test.jpg
```

For SSH without a local display, use `rpicam-hello --nopreview --timeout 5000` and inspect the saved still separately.
The default preview does not support ordinary X forwarding.
- [ ] For Python control of the CSI camera, use Picamera2, installed through apt.
Do not assume OpenCV `VideoCapture(0)` supports the Camera Module 3.

## Prepare WasteSort after camera testing

These project-specific steps remain untested on the Pi.

- [ ] Clone the team repository and enter it.
- [ ] Create a virtual environment that can import the system Picamera2 and OpenCV packages:

```sh
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'from picamera2 import Picamera2; print("Picamera2 import OK")'
```

- [ ] Record Python, PyTorch, torchvision, OpenCV, and OS versions.
Verify PyTorch/torchvision wheels support the installed 64-bit OS and Python; log any installation failure before proceeding.
- [ ] Obtain the team's trained checkpoint, record its SHA-256, and place it at `models/resnet18_realwaste.pt`.
Do not replace it with ImageNet weights or retrain as a setup workaround.
- [ ] Run `python ml/predict.py --image ~/camera-test.jpg` to check the file-based prediction path.
Record the result and device before attempting camera integration.
- [ ] Plan a separate Picamera2 capture adapter with the team before using a CSI camera for WasteSort.
The laptop `ml/live_demo.py` currently targets webcam 0.
- [ ] Keep Pi latency and physical item accuracy pending until measured on this actual setup.
Record resolution, lighting, model checksum, warm-up, ten CPU samples, mean, and range in the camera requirements document.
