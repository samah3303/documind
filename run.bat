@echo off
echo ============================================
echo   DocuMind - Enterprise RAG System
echo ============================================
echo.

REM Check for .env
if not exist .env (
    echo [WARNING] .env file not found!
    echo Copy .env.example to .env and fill in your API keys.
    echo.
    choice /C YN /M "Continue without .env"
    if errorlevel 2 exit /b 1
)

REM Create directories
if not exist chroma_db mkdir chroma_db
if not exist uploads mkdir uploads

echo [1/2] Starting FastAPI backend...
start "DocuMind Backend" cmd /k "cd backend && uv pip install -r requirements.txt 2>nul & pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend to start
echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo [2/2] Starting Streamlit frontend...
start "DocuMind Frontend" cmd /k "cd frontend && pip install -r requirements.txt && streamlit run app.py"

echo.
echo ============================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:8501
echo   API Docs: http://localhost:8000/docs
echo ============================================
echo.
echo Close the terminal windows to stop the servers.
pause
