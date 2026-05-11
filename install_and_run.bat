@echo off
echo  Starting... ~_~

IF NOT EXIST venv (
   echo Creating virtual environment...
   python -m venv venv
) 

call venv\Scripts\activate


echo Installing runtime requirements...
pip install -r requirements_runtime.txt

echo Starting application...
python ui/app.py

pause