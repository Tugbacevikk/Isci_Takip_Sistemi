@echo off
chcp 65001 >nul
title Isci Takip Sistemi

REM --- UTF-8 Turkce Karakter Destegi ---
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo.
echo ============================================================
echo   ISCI TAKIP SISTEMI
echo   Tarayicida acin: http://localhost:5000
echo ============================================================
echo.

cd /d "%~dp0"
python web/app.py

if errorlevel 1 (
    echo HATA! Yukaridaki mesaji inceleyin.
    pause
)
