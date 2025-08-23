@echo off
echo Starting RadiaCode Isotope Detector...
echo.
echo Make sure you have all trained models in the ../models/ directory
echo.

REM Activate virtual environment if it exists
if exist "..\.venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\.venv\Scripts\activate.bat
)

REM Start Streamlit app
streamlit run streamlit_isotope_detector.py --server.port 8501 --server.address localhost

pause
