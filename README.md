# RapidVision

**RapidVision** is a real-time object detection tool powered by the PP-YOLOE deep learning model and the COCO object class dataset. It supports switching between multiple video sources and is built for responsive, flexible object recognition.

RapidVision is currently focused on fast and accurate detection of common object categories (e.g., people, vehicles, animals, household items) with a lightweight and modular Python architecture.

> **Planned features** include hardware-accelerated inference (e.g., NVIDIA Jetson, Intel NPU) and support for security surveillance tasks such as frame recording, event logging, and smart detection triggers.

---

## Installation

Follow these steps to set up the RapidVision environment and dependencies.

1. **Clone the repository**

   ```bash
   git clone https://github.com/Nardo45/RapidVision.git
   cd RapidVision
   ```

2. **Create and activate the Conda environment**

   ```bash
   conda env create -f environment.yaml
   conda activate rapidvision_env
   ```

3. **Install additional Python dependencies**
   If you add new dependencies, install them with:

   ```bash
   pip install <package-name>
   ```

4. **Run RapidVision**

   ```bash
   python main.py
   ```

---

## Usage

* **Switch cameras**: Use the Left/Right arrow keys to cycle through connected cameras.
* **Open Settings**: Press the `K` key to open the settings menu.
* **Camera Profiles**: In the settings menu, use "Camera Profiles" to select or create presets.
* **Camera Calibration**: In the settings menu, use "Calibrate Camera" to create measurements by moving a checkerboard across the camera. These measurements can slightly improve distance estimation.

---

## License and Attribution

RapidVision is licensed under the [LGPL-3.0 License](LICENSE.md).

This project includes third-party code under separate licenses:

- [PPYOLOE_pytorch](https://github.com/Nioolek/PPYOLOE_pytorch) (Apache 2.0)  
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache 2.0)  

See [THIRD_PARTY_LICENSES.md](third_party/THIRD_PARTY_LICENSES.md) for full details.
