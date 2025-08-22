#!/usr/bin/env python3
"""
Установка зависимостей для веб-версии в Google Colab
"""

import subprocess
import sys
import os

def install_packages():
    """Установка необходимых пакетов"""
    packages = [
        "gradio",
        "TTS==0.21.0",
        "torch==2.0.1",
        "torchaudio==2.0.1",
        "transformers==4.30.0",
        "soundfile",
        "librosa",
        "numpy",
        "matplotlib"
    ]
    
    print("🔄 Установка пакетов для веб-версии...")
    
    for package in packages:
        print(f"📦 Устанавливаем {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} установлен")
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка установки {package}: {e}")
    
    print("✅ Все пакеты установлены!")

def setup_colab():
    """Настройка среды Colab"""
    print("⚙️ Настройка Google Colab...")
    
    # Проверяем, что мы в Colab
    try:
        import google.colab
        print("✅ Google Colab обнаружен")
    except ImportError:
        print("⚠️ Не в Google Colab, но продолжаем...")
    
    # Установка пакетов
    install_packages()
    
    print("""
🎉 НАСТРОЙКА ЗАВЕРШЕНА!

Теперь запустите веб-версию:
    python voice_cloner_web.py

Или в Colab:
    !python voice_cloner_web.py

После запуска появится ссылка вида:
    https://xxxxx.gradio.live

Перейдите по ней для использования интерфейса!
""")

if __name__ == "__main__":
    setup_colab()

