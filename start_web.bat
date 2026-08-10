@echo off
chcp 65001 >nul
title Isci Takip Sistemi - Web Sunucusu
color 0A

REM --- UTF-8 Turkce Karakter Destegi ---
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo.
echo ============================================================
echo   ISCI TAKIP SISTEMI - Web Sunucusu
echo   Tarayicida acin: http://localhost:5000
echo ============================================================
echo.
echo Bagimliliklar kontrol ediliyor...
pip install -r requirements_web.txt -q
echo.
echo Web sunucusu baslatiliyor...
cd /d "%~dp0"
python web/app.py

if errorlevel 1 (
    echo.
    echo HATA: Uygulama baslatamadi!
    echo Yukaridaki hata mesajini inceleyin.
    pause
)
