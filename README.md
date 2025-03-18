## Description:
Real time recording and plotting platform for TI-xWR6843-based radars (ODS/AoP)
- 100 Hz 
- 1 Antenna
- 1 Chirp

## Features:
- Presence Detection
- Motion Detection **(To be added)**
- Respiration Analysis (signal, rate, patterns)
- Heartbeat Analysis (signal, rate)
- Support for reference devices (Polar H10, Vernier respiration Belt, Witmotion, Camera, Others)
- Graphical User Interface
- Realtime plotting
- Data recording (Excel radar and reference and MP4 for video from camera)
- Customizable recording protocol for experiments

## Requirements:
- Project was tested using Windows 10 Python 3.9.x 
- Use TI-xWR6843-based radar (ODS/AoP)
- Check if radar UART driver is installed from Device Manager tool (install from "firmware/drivers" folder if necessary)
- Install dependencies using "pip install -r requirements.txt"
- Install kivy garden dependencies: "garden install iconfonts" and "garden install matplotlib"

## Usage:
- Modify/Update "User Setting" in "config.py" according to your requirements
- Run using "python main.py"
- .venv\Scripts\activate

## Roadmap:
- Add Eyeblink detection option
- Add Actigraph for sleep scoring
- Add emotion recognition