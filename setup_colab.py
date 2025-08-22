#!/usr/bin/env python3
"""
Скрипт для настройки Google Colab для работы с GUI приложениями
Запустите этот скрипт перед запуском voice_cloner_xtts_v2.py
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Выполнить команду и показать результат"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - успешно")
            if result.stdout.strip():
                print(f"   Вывод: {result.stdout.strip()}")
        else:
            print(f"❌ {description} - ошибка")
            if result.stderr.strip():
                print(f"   Ошибка: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {description} - исключение: {e}")
        return False

def main():
    print("🚀 Настройка Google Colab для GUI приложений")
    print("=" * 50)
    
    # Проверяем, что мы в Google Colab
    if 'google.colab' not in sys.modules:
        print("⚠️ Этот скрипт предназначен для Google Colab")
        print("На обычном компьютере GUI должен работать без дополнительной настройки")
        return
    
    # Обновляем пакеты
    run_command("apt-get update", "Обновление списка пакетов")
    
    # Устанавливаем Xvfb (виртуальный дисплей)
    run_command("apt-get install -y xvfb", "Установка Xvfb")
    
    # Устанавливаем pyvirtualdisplay
    run_command("pip install pyvirtualdisplay", "Установка pyvirtualdisplay")
    
    # Проверяем установку
    print("\n🔍 Проверка установки...")
    
    # Проверяем Xvfb
    xvfb_ok = run_command("which Xvfb", "Проверка Xvfb")
    
    # Проверяем pyvirtualdisplay
    try:
        import pyvirtualdisplay
        print("✅ pyvirtualdisplay установлен")
        pyvirtualdisplay_ok = True
    except ImportError:
        print("❌ pyvirtualdisplay не установлен")
        pyvirtualdisplay_ok = False
    
    print("\n" + "=" * 50)
    if xvfb_ok and pyvirtualdisplay_ok:
        print("🎉 Настройка завершена успешно!")
        print("Теперь вы можете запустить voice_cloner_xtts_v2.py")
    else:
        print("⚠️ Настройка завершена с ошибками")
        print("Попробуйте перезапустить runtime в Colab и выполнить скрипт снова")
    
    print("\n💡 Для запуска основного приложения выполните:")
    print("   python voice_cloner_xtts_v2.py")

if __name__ == "__main__":
    main()
