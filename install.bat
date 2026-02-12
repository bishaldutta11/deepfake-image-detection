@echo off
echo Installing Deepfake Detection Dependencies...
python -m pip install --upgrade pip
pip install flask==2.3.3 flask-cors==4.0.0 numpy==1.24.3 pillow==10.0.1 opencv-python==4.8.1.78 tensorflow==2.16.2
echo.
echo Installation complete!
echo Starting the application...
python app.py
pause
