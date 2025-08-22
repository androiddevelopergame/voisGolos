#!/usr/bin/env python3
"""
Скрипт для автоматического исправления проблем с pyaudio
"""

import subprocess
import sys
import platform

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
    print("🔧 Исправление проблем с PyAudio")
    print("=" * 50)
    
    system = platform.system().lower()
    print(f"🖥️ Обнаружена система: {system}")
    
    # Удаляем старую версию pyaudio
    print("\n🗑️ Удаление старой версии PyAudio...")
    run_command("pip uninstall pyaudio -y", "Удаление pyaudio")
    
    # Устанавливаем зависимости в зависимости от системы
    if system == "linux":
        print("\n🐧 Установка зависимостей для Linux...")
        run_command("sudo apt-get update", "Обновление пакетов")
        run_command("sudo apt-get install -y portaudio19-dev python3-pyaudio", "Установка portaudio")
        
    elif system == "darwin":  # macOS
        print("\n🍎 Установка зависимостей для macOS...")
        run_command("brew install portaudio", "Установка portaudio через brew")
        
    elif system == "windows":
        print("\n🪟 Установка зависимостей для Windows...")
        # Пробуем установить pipwin если его нет
        try:
            import pipwin
            print("✅ pipwin уже установлен")
        except ImportError:
            run_command("pip install pipwin", "Установка pipwin")
    
    # Устанавливаем pyaudio
    print("\n📦 Установка PyAudio...")
    if system == "windows":
        success = run_command("pipwin install pyaudio", "Установка pyaudio через pipwin")
        if not success:
            print("🔄 Пробуем обычную установку...")
            run_command("pip install pyaudio", "Установка pyaudio через pip")
    else:
        run_command("pip install pyaudio", "Установка pyaudio")
    
    # Проверяем установку
    print("\n🔍 Проверка установки PyAudio...")
    try:
        import pyaudio
        print("✅ PyAudio импортирован успешно")
        
        # Проверяем доступные форматы
        pa_attrs = [attr for attr in dir(pyaudio) if attr.startswith('pa')]
        print(f"📋 Доступные форматы: {pa_attrs}")
        
        # Проверяем конкретные форматы
        if hasattr(pyaudio, 'paInt16'):
            print("✅ paInt16 - доступен")
        else:
            print("❌ paInt16 - НЕ доступен")
            
        if hasattr(pyaudio, 'paInt32'):
            print("✅ paInt32 - доступен")
        else:
            print("❌ paInt32 - НЕ доступен")
            
        # Тестируем инициализацию
        try:
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            print(f"✅ PyAudio инициализирован, найдено устройств: {device_count}")
            p.terminate()
        except Exception as e:
            print(f"❌ Ошибка инициализации PyAudio: {e}")
            
    except ImportError as e:
        print(f"❌ Ошибка импорта PyAudio: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка при работе с PyAudio: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Исправление PyAudio завершено!")
    print("💡 Теперь попробуйте запустить основное приложение:")
    print("   python voice_cloner_xtts_v2.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
