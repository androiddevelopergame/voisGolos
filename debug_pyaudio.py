#!/usr/bin/env python3
"""
Скрипт для диагностики pyaudio и его доступных атрибутов
"""

try:
    import pyaudio
    print("✅ pyaudio импортирован успешно")
    
    # Выводим все атрибуты, начинающиеся с 'pa'
    pa_attrs = [attr for attr in dir(pyaudio) if attr.startswith('pa')]
    print(f"📋 Доступные форматы pyaudio: {pa_attrs}")
    
    # Проверяем конкретные форматы
    formats_to_check = ['paInt16', 'paInt32', 'paFloat32', 'paInt8', 'paUInt8']
    for fmt in formats_to_check:
        if hasattr(pyaudio, fmt):
            print(f"✅ {fmt} - доступен")
        else:
            print(f"❌ {fmt} - НЕ доступен")
    
    # Показываем все атрибуты pyaudio
    print(f"\n📋 Все атрибуты pyaudio: {dir(pyaudio)}")
    
except ImportError as e:
    print(f"❌ Ошибка импорта pyaudio: {e}")
except Exception as e:
    print(f"❌ Ошибка при работе с pyaudio: {e}")
