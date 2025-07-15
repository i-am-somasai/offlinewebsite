Offline AI Voice & Video Assistant — Setup Guide

Prerequisites:
Ensure you have the following installed:
•	Python 3.10 (Recommended because speech_recognition and some pyaudio dependencies may break on 3.11+.)
•	Git
•	A modern browser (Microsoft Edge or Google Chrome recommended)

Step-by-Step Project Setup:
1. Clone the Repository:
Download all necessary project files from the GitHub repository:
🔗 i-am-somasai/offlinewebsite

2. Download marked folder from:
https://github.com/markedjs/marked/releases
Place the entire marked folder inside the static directory.

3. Set Up Python Virtual Environment:
Open Command Prompt and navigate to your project directory:
cd path\to\your_project
Then create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate # On Windows


4. Install Python Dependencies:
Run the following to install required packages:
pip install -r requirements.txt

5. Launch the Application:
You can launch the site using either:
python launcher.py -------- Recommended
OR
uvicorn main:app --reload --port 8080 ---------- Highly recommended
OR 
fastapi run main.py --port 8080
Replace 8080 with your preferred port.
Note:
If you face issues with numpy, fix it by:
pip uninstall numpy
pip install numpy==1.26.4

6. Installing Piper (for TTS):
1.	Download the latest stable Piper build from:
🔗 https://github.com/rhasspy/piper/releases
2.	Extract the ZIP file.
3.	Inside the extracted folder, locate the executable (piper binary or .exe).
4.	Copy the folder path and add it to your system's Environment Variables (System PATH).

7. Downloading Voices:
Visit the Piper voices list:
🔗 https://github.com/rhasspy/piper/blob/master/VOICES.md

Download:
•	A .onnx file (model)
•	A corresponding .json config file
Important:
•	For .json files, open in browser and save as .json via Ctrl+S.
•	Place these files in the Piper directory or any directory of your choice.
Update the path in main.py to match the folder where the voice files are stored.

8. Test Piper Installation:
Before testing, ensure you edit config and model paths in test_piper.py.
Then run:
python test_piper.py

9. Voice Dropdown Setup (Optional):
To enable dropdown voice selection in your app:
•	Place all downloaded .onnx and .json files into one folder.
•	Copy the path of this folder.
•	Update this path in main.py to populate the dropdown menu.

10. Connect Mi Smart Speaker:
1.	Reset the speaker by holding both volume buttons.
2.	Open Google Home app on your mobile.
3.	Follow setup instructions, connect to Wi-Fi.
4.	Enable pairing mode by saying:
"Ok Google, go to Bluetooth pairing mode"
5.	Now connect the speaker via Bluetooth settings.


11. Troubleshooting:
•	ModuleNotFoundError? → Check if virtual environment is activated.
•	Piper not speaking? → Ensure the .onnx and .json voice files match and paths are correct in test_piper.py.
•	FastAPI not recognized? → Use uvicorn main:app --reload --port 8080 instead of fastapi run (FastAPI doesn’t have a run command itself).

12. Check the port accessibility: 
•	If this will run offline but open ports:
o	Ensure your firewall or antivirus doesn’t block port 8080
o	If 8080 is selected as the port, then check no other service is using that port.

13. Organize the Project Structure:
your_project/
└── backend/
    ├── static/ ----------> Contains frontend files (HTML, CSS, JS).
    │   ├── marked/
    │   ├── index.html
    │   ├── style.css
    │   ├── script.js
    │   └── manifest.json
    ├── main.py --------> FastAPI backend entry point.
    ├── launcher.py ----> GUI launcher or alternate entry.
    ├── test_piper.py ---> TTS voice testing script.
    └── requirements.txt

