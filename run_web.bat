@echo off
chcp 65001
echo Запуск веб-версии клонировщика голоса XTTS v2...
echo.
echo После запуска откройте в браузере:
echo http://127.0.0.1:7860
echo.
py -3.11 voice_cloner_web.py
pause
