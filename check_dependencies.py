#!/usr/bin/env python3
"""
Скрипт для проверки зависимостей приложения клонирования голоса
Запустите этот скрипт для диагностики проблем с установкой
"""

import sys
import importlib

def check_module(module_name, package_name=None, install_command=None):
    """Проверить установку модуля"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - установлен")
        return True
    except ImportError:
        print(f"❌ {module_name} - НЕ установлен")
        if install_command:
            print(f"   Установите: {install_command}")
        elif package_name:
            print(f"   Установите: pip install {package_name}")
        return False

def main():
    print("🔍 Проверка зависимостей для клонирования голоса")
    print("=" * 50)
    
    # Основные зависимости
    dependencies = [
        ("tkinter", None, None),  # Обычно входит в Python
        ("torch", "torch", "pip install torch torchaudio"),
        ("TTS", "TTS", "pip install TTS"),
        ("pyaudio", "pyaudio", "pip install pyaudio (или pipwin install pyaudio на Windows)"),
        ("soundfile", "soundfile", "pip install soundfile"),
        ("librosa", "librosa", "pip install librosa"),
        ("matplotlib", "matplotlib", "pip install matplotlib"),
        ("numpy", "numpy", "pip install numpy"),
        ("pyttsx3", "pyttsx3", "pip install pyttsx3"),
        ("scipy", "scipy", "pip install scipy"),
        ("transformers", "transformers", "pip install transformers"),
        ("accelerate", "accelerate", "pip install accelerate"),
    ]
    
    # Проверяем каждую зависимость
    all_ok = True
    for module, package, install_cmd in dependencies:
        if not check_module(module, package, install_cmd):
            all_ok = False
    
    # Специальная проверка для pyaudio
    print("\n🔍 Дополнительная проверка PyAudio...")
    try:
        import pyaudio
        # Проверяем доступные форматы
        formats = [attr for attr in dir(pyaudio) if attr.startswith('pa')]
        print(f"   Доступные форматы: {', '.join(formats[:5])}...")
        
        # Проверяем наличие микрофонов
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        print(f"   Найдено аудио устройств: {device_count}")
        
        # Ищем устройства ввода
        input_devices = []
        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append(info['name'])
            except:
                pass
        
        if input_devices:
            print(f"   Устройства ввода: {', '.join(input_devices[:3])}...")
        else:
            print("   ⚠️ Устройства ввода не найдены")
        
        p.terminate()
        
    except Exception as e:
        print(f"   ❌ Ошибка проверки PyAudio: {e}")
        all_ok = False
    
    # Проверка для Google Colab
    if 'google.colab' in sys.modules:
        print("\n🔍 Проверка Google Colab...")
        colab_deps = [
            ("pyvirtualdisplay", "pyvirtualdisplay", "pip install pyvirtualdisplay"),
        ]
        
        for module, package, install_cmd in colab_deps:
            if not check_module(module, package, install_cmd):
                all_ok = False
        
        # Проверяем Xvfb
        import subprocess
        try:
            result = subprocess.run(['which', 'Xvfb'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Xvfb - установлен")
            else:
                print("❌ Xvfb - НЕ установлен")
                print("   Установите: !apt-get install -y xvfb")
                all_ok = False
        except:
            print("⚠️ Не удалось проверить Xvfb")
    
    print("\n" + "=" * 50)
    if all_ok:
        print("🎉 Все зависимости установлены! Приложение готово к запуску.")
        print("💡 Запустите: python voice_cloner_xtts_v2.py")
    else:
        print("⚠️ Некоторые зависимости не установлены.")
        print("💡 Установите недостающие пакеты и запустите проверку снова.")
        print("💡 Или используйте: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
