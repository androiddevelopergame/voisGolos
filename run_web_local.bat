@echo off
chcp 65001
echo 🏠 Запуск веб-версии в ЛОКАЛЬНОМ режиме...
echo.
echo 🔒 Доступ только с этого компьютера
echo 📍 Адрес: http://127.0.0.1:7860
echo.
py -3.11 voice_cloner_web.py
pause
