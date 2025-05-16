# MocapV2 - Computer Vision Motion Capture

![License: MIT Personal Use](https://img.shields.io/badge/License-MIT%20Personal%20Use-blue.svg)

MocapV2 is a Python-based motion capture system that utilizes computer vision techniques, primarily leveraging OpenCV, to detect and track infrared (IR) markers attached to objects in real-time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

MocapV2 aims to provide an accessible motion capture solution using readily available hardware, such as standard webcams. The system analyzes video input frame by frame, identifies IR markers, and estimates their 3D positions.

Originally developed for tracking studio cameras, MocapV2 can also be used for general motion capture applications. The modular architecture facilitates easy extension and integration into larger projects.

## Features

- **Real-time Tracking:** Detects and tracks IR markers from a video stream.
- **OpenCV Integration:** Uses OpenCV for video capture, image processing, and tracking.
- **Visual Feedback:** The GUI displays camera feeds and allows real-time parameter adjustments.
- **Realtime Socket Communication:** The calculated object transformations can be transmitted via sockets, enabling integration with applications like Unity 3D, Unreal Engine, and Blender.
- **Modular Codebase:** Organized structure for ease of development and scalability.

## Tech Stack

- **Language:** Python 3.x
- **Core Libraries:**
  - OpenCV (opencv-python) - Video capture, image processing, and display.
  - NumPy - Efficient numerical computations.

## Installation

Follow these steps to set up the project:

### 1. Clone the Repository
```bash
git clone https://github.com/RashmikaDushan/MocapV2.git
cd MocapV2
```

### 2. Create a Virtual Environment (Recommended)

Using `venv`:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Using `conda`:
```bash
conda create -n mocapv2 python=3.9  # Adjust Python version if needed
conda activate mocapv2
```

### 3. Install Dependencies
```bash
pip install opencv-python numpy
```

**Note:** Depending on your OS and Python version, OpenCV might require additional system dependencies.

## Usage

To run the real-time motion capture demo:

1. Ensure that at least two webcams are connected and accessible.
2. Navigate to the project's root directory:
   ```bash
   cd MocapV2
   ```
3. Activate the virtual environment (if not already activated).
4. Run the main application script:
   ```bash
   python app.py
   ```

A window will open, displaying the webcam feed with settings in real-time. Press `Esc` to close the program.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository on GitHub.
2. Clone your fork:
   ```bash
   git clone https://github.com/YourUsername/MocapV2.git
   ```
3. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name  # or bugfix/issue-description
   ```
4. Make changes and commit:
   ```bash
   git commit -am 'Add feature X'
   ```
5. Push changes:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request (PR) on the original repository.

Ensure your code follows best practices and update documentation if needed. You can also report bugs or suggest features by opening an issue on GitHub.

## License

This project is licensed under the **MIT Personal Use License**.

You may use and modify this software for **personal, non-commercial purposes only**.

For full details, refer to the [LICENSE](LICENSE) file.

## Acknowledgements

- Inspired by [jyjblrd/Low-Cost-Mocap](https://github.com/jyjblrd/Low-Cost-Mocap.git), with portions of the code adapted from it.
- Built using [OpenCV](https://opencv.org/) for essential computer vision tasks.

