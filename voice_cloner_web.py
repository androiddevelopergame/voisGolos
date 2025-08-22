#!/usr/bin/env python3
"""
Веб-версия клонировщика голоса для Google Colab
Использует Gradio для создания веб-интерфейса
"""

import gradio as gr
import tempfile
import os
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

# Импорт TTS
try:
    from TTS.api import TTS
    print("✅ TTS импортирован успешно")
except ImportError as e:
    print(f"❌ Ошибка импорта TTS: {e}")
    print("💡 Установите: pip install TTS")

class VoiceClonerWeb:
    def __init__(self):
        self.xtts_model = None
        self.init_model()
    
    def init_model(self):
        """Инициализация модели XTTS v2"""
        try:
            print("🔄 Загрузка XTTS v2...")
            
            # Исправление для PyTorch 2.6
            try:
                from torch.serialization import add_safe_globals
                from TTS.tts.configs.xtts_config import XttsConfig
                safe_classes = [XttsConfig]
                add_safe_globals(safe_classes)
                print("✅ PyTorch safe globals добавлены")
            except Exception as e:
                print(f"⚠️ Предупреждение safe globals: {e}")
            
            # Загрузка модели
            self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("✅ XTTS v2 загружена успешно!")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.xtts_model = None
    
    def clone_voice(self, text, voice_file, language="ru", temperature=0.7, speed=1.0):
        """Клонирование голоса"""
        if not self.xtts_model:
            return None, "❌ Модель XTTS v2 не загружена!"
        
        if not voice_file:
            return None, "❌ Загрузите файл с голосом!"
        
        if not text.strip():
            return None, "❌ Введите текст для озвучки!"
        
        try:
            print(f"🎯 Генерация голоса для текста: {text[:50]}...")
            
            # Создаем временный файл для результата
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # Генерация с клонированием голоса
            self.xtts_model.tts_to_file(
                text=text,
                speaker_wav=voice_file,
                language=language,
                file_path=output_path,
                speed=speed
            )
            
            print(f"✅ Голос сгенерирован: {output_path}")
            return output_path, f"✅ Голос успешно сгенерирован! Длина текста: {len(text)} символов"
            
        except Exception as e:
            error_msg = f"❌ Ошибка генерации: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def get_voice_info(self, voice_file):
        """Получить информацию о голосовом файле"""
        if not voice_file:
            return "Файл не загружен"
        
        try:
            # Загрузка аудио для анализа
            y, sr = sf.read(voice_file)
            duration = len(y) / sr
            
            return f"""📊 Информация о голосовом файле:
• Длительность: {duration:.1f} секунд
• Частота дискретизации: {sr} Гц
• Каналов: {'моно' if len(y.shape) == 1 else 'стерео'}
• Размер файла: {os.path.getsize(voice_file) / 1024 / 1024:.1f} МБ

{'✅ Файл подходит для клонирования' if duration >= 10 else '⚠️ Рекомендуется файл длиннее 10 секунд'}"""
            
        except Exception as e:
            return f"❌ Ошибка анализа файла: {str(e)}"

def create_interface():
    """Создание веб-интерфейса"""
    cloner = VoiceClonerWeb()
    
    # Пример текста
    example_text = """Привет! Это пример текста для озвучки вашим клонированным голосом.

Программа использует XTTS v2 для клонирования голоса с поддержкой русского языка.
Вы можете вставить любой текст на русском языке, и он будет озвучен качественно.

Рекомендации для лучшего качества:
• Используйте качественную запись голоса (10+ секунд)
• Говорите четко и без фонового шума
• Текст должен быть на русском языке"""
    
    with gr.Blocks(title="🎤 Клонирование Голоса XTTS v2", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎤 Клонирование Голоса с XTTS v2
        
        ### 🇷🇺 Русскоязычный клонировщик голоса на основе XTTS v2
        
        **Инструкция:**
        1. Загрузите файл с голосом (WAV, MP3, FLAC)
        2. Введите текст для озвучки
        3. Настройте параметры (по желанию)
        4. Нажмите "Генерировать голос"
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Загрузка голоса")
                
                voice_file = gr.Audio(
                    label="Файл с голосом для клонирования",
                    type="filepath",
                    sources=["upload"]
                )
                
                voice_info = gr.Textbox(
                    label="Информация о файле",
                    value="Загрузите файл для анализа",
                    interactive=False,
                    max_lines=8
                )
                
                gr.Markdown("### ⚙️ Настройки")
                
                language = gr.Dropdown(
                    choices=["ru", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"],
                    value="ru",
                    label="Язык",
                    info="Выберите язык текста"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Температура (креативность)",
                    info="Меньше = предсказуемо, больше = креативно"
                )
                
                speed = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Скорость речи",
                    info="Меньше = медленнее, больше = быстрее"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Текст для озвучки")
                
                text_input = gr.Textbox(
                    label="Введите текст",
                    value=example_text,
                    lines=12,
                    max_lines=20,
                    placeholder="Введите текст на русском языке..."
                )
                
                generate_btn = gr.Button(
                    "🎯 Генерировать голос",
                    variant="primary",
                    size="lg"
                )
                
                status = gr.Textbox(
                    label="Статус",
                    value="Готов к работе",
                    interactive=False
                )
                
                result_audio = gr.Audio(
                    label="Результат",
                    type="filepath"
                )
        
        # События
        voice_file.change(
            fn=cloner.get_voice_info,
            inputs=[voice_file],
            outputs=[voice_info]
        )
        
        generate_btn.click(
            fn=cloner.clone_voice,
            inputs=[text_input, voice_file, language, temperature, speed],
            outputs=[result_audio, status]
        )
        
        gr.Markdown("""
        ### 💡 Советы для лучшего качества:
        
        **Файл с голосом:**
        - Длительность: 10-60 секунд
        - Качество: без шумов и эха
        - Формат: WAV, MP3, FLAC
        - Четкая речь на русском языке
        
        **Текст:**
        - Используйте русский язык
        - Избегайте специальных символов
        - Длина: от 10 до 500 слов
        
        **Настройки:**
        - Температура 0.7 - оптимальная для большинства случаев
        - Скорость 1.0 - нормальная скорость речи
        """)
    
    return interface

def main():
    """Запуск веб-интерфейса"""
    print("🚀 Запуск веб-интерфейса клонирования голоса...")
    
    interface = create_interface()
    
    # Запуск с публичной ссылкой для Colab
    interface.launch(
        share=True,  # Создает публичную ссылку
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )

if __name__ == "__main__":
    main()

