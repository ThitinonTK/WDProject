@echo off
REM Change directory to where your virtual environment is located
cd /d "E:\Yoom\"

REM Activate the virtual environment
call env\Scripts\activate

REM Change directory to where your script is located
cd /d "E:\Yoom\GUI\"

REM Run the Python script
python app_5_parameter.py

REM Deactivate the virtual environment
REM deactivate

REM Pause to keep the command window open
REM pause
