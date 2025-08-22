import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pyaudio
import wave
import time
import pyttsx3
import platform
import sys

# Проверяем, запущены ли мы в среде без дисплея (Google Colab, сервер и т.д.)
def setup_display():
    """Настройка дисплея для работы в различных средах"""
    # Проверяем наличие дисплея
    has_display = os.environ.get('DISPLAY') is not None
    
    # Проверяем, запущены ли мы в Google Colab
    is_colab = 'google.colab' in sys.modules
    
    # Если нет дисплея или мы в Colab, пытаемся создать виртуальный
    if not has_display or is_colab:
        try:
            from pyvirtualdisplay import Display
            print("🖥️ Обнаружена среда без дисплея, запускаем виртуальный дисплей...")
            
            # Устанавливаем переменную DISPLAY перед созданием виртуального дисплея
            if not os.environ.get('DISPLAY'):
                os.environ['DISPLAY'] = ':99'
            
            display = Display(visible=0, size=(1920, 1080))
            display.start()
            
            # Проверяем, что дисплей действительно запущен
            import subprocess
            try:
                result = subprocess.run(['xdpyinfo'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print("✅ Виртуальный дисплей успешно запущен и работает")
                    return True
                else:
                    print("⚠️ Виртуальный дисплей создан, но не отвечает")
                    return False
            except:
                print("✅ Виртуальный дисплей создан (xdpyinfo недоступен)")
                return True
                
        except ImportError:
            print("⚠️ pyvirtualdisplay не установлен. Для работы в Colab установите: pip install pyvirtualdisplay")
            print("🔄 Продолжаем без виртуального дисплея...")
            return False
        except Exception as e:
            print(f"⚠️ Ошибка запуска виртуального дисплея: {e}")
            print("🔄 Продолжаем без виртуального дисплея...")
            return False
    else:
        print("✅ Дисплей обнаружен, используем стандартный режим")
        return True

# Запускаем настройку дисплея
display_ready = setup_display()

# Если дисплей не готов, устанавливаем переменную DISPLAY вручную
if not display_ready and not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':99'
    print("🔧 Установлена переменная DISPLAY=:99")

# Исправление для PyTorch 2.6 - добавляем все необходимые классы
try:
    from torch.serialization import add_safe_globals
    
    # Основные классы для XTTS v2
    safe_classes = []
    
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        safe_classes.append(XttsConfig)
    except: pass
    
    try:
        from TTS.tts.models.xtts import XttsAudioConfig
        safe_classes.append(XttsAudioConfig)
    except: pass
    
    try:
        from TTS.tts.models.xtts import XttsArgs
        safe_classes.append(XttsArgs)
    except: pass
    
    try:
        from TTS.config.shared_configs import BaseDatasetConfig
        safe_classes.append(BaseDatasetConfig)
    except: pass
    
    try:
        from TTS.tts.configs.shared_configs import CharactersConfig
        safe_classes.append(CharactersConfig)
    except: pass
    
    try:
        from TTS.vocoder.configs.hifigan_config import HifiganConfig
        safe_classes.append(HifiganConfig)
    except: pass
    
    if safe_classes:
        add_safe_globals(safe_classes)
        print(f"✅ PyTorch 2.6 safe globals добавлены: {len(safe_classes)} классов")
    else:
        print("⚠️ Не удалось найти классы для safe globals")
        
except Exception as e:
    print(f"⚠️ Ошибка добавления safe globals: {e}")
    pass

from TTS.api import TTS

class VoiceClonerXTTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Клонирование Голоса - XTTS v2 + Windows TTS")
        self.root.geometry("1200x1000")
        
        # Переменные
        self.voice_file_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.standard_output_path = tk.StringVar()
        self.xtts_model = None  # XTTS v2 для клонирования
        self.windows_tts = None  # Системный TTS Windows
        self.is_processing = False
        self.is_recording = False
        self.recording_thread = None
        
        # Аудио параметры - улучшенные для качества
        self.CHUNK = 2048  # Увеличили размер чанка для лучшего качества
        self.FORMAT = pyaudio.paInt16  # Стандартный формат для совместимости
        self.CHANNELS = 1
        self.RATE = 48000  # Увеличили частоту дискретизации для лучших высоких частот
        
        # Инициализация PyAudio с обработкой ошибок
        try:
            self.audio = pyaudio.PyAudio()
            print("✅ PyAudio инициализирован успешно")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации PyAudio: {e}")
            print("💡 Для записи голоса установите pyaudio: pip install pyaudio")
            self.audio = None
        
        # Создание интерфейса
        self.create_widgets()
        
        # Инициализация моделей
        self.init_models()
        
    def create_widgets(self):
        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка весов
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Клонирование Голоса - XTTS v2 + Windows TTS", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Информация о версии
        info_label = ttk.Label(main_frame, text="XTTS v2 (клонирование + русский) + Системный TTS Windows", 
                              font=("Arial", 10), foreground="blue")
        info_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Важное примечание
        note_label = ttk.Label(main_frame, 
                              text="🇷🇺 РУССКИЙ TTS: XTTS v2 с клонированием голоса + Windows TTS для сравнения", 
                              font=("Arial", 10, "bold"), foreground="green")
        note_label.grid(row=2, column=0, columnspan=3, pady=(0, 20))
        
        # Секция записи голоса
        recording_frame = ttk.LabelFrame(main_frame, text="Запись голоса через микрофон", padding="10")
        recording_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        recording_frame.columnconfigure(1, weight=1)
        
        # Кнопки записи
        ttk.Label(recording_frame, text="Запись голоса:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        button_frame = ttk.Frame(recording_frame)
        button_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.record_button = ttk.Button(button_frame, text="🎤 Начать запись", 
                                       command=self.start_recording, style="Accent.TButton")
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Остановить запись", 
                                     command=self.stop_recording, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Индикатор записи
        self.recording_indicator = ttk.Label(button_frame, text="", font=("Arial", 12))
        self.recording_indicator.pack(side=tk.LEFT, padx=(10, 0))
        
        # Таймер записи
        self.recording_timer = tk.StringVar(value="00:00")
        ttk.Label(button_frame, textvariable=self.recording_timer, 
                 font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=(10, 0))
        
        # Информация о записи
        recording_info = """🎤 ИНСТРУКЦИЯ ПО ЗАПИСИ ВЫСОКОГО КАЧЕСТВА:
 • Используйте качественный микрофон (не встроенный)
 • Говорите четко и естественно в микрофон
 • Рекомендуется 60-120 секунд разнообразной речи
 • Избегайте фонового шума и эха
 • Говорите именно на русском языке
 • Расстояние до микрофона: 10-20 см
 • Говорите с нормальной громкостью"""
        
        ttk.Label(recording_frame, text=recording_info, 
                 font=("Arial", 9), foreground="gray", justify=tk.LEFT).grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        # Секция загрузки файла
        file_frame = ttk.LabelFrame(main_frame, text="🎵 ЗАГРУЗИТЬ ГОТОВЫЙ ФАЙЛ С ГОЛОСОМ", padding="15")
        file_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        file_frame.columnconfigure(1, weight=1)
        
        # БОЛЬШАЯ ЗАМЕТНАЯ КНОПКА СВЕРХУ!
        big_button = tk.Button(file_frame, text="📁 ВЫБРАТЬ ФАЙЛ С ГОЛОСОМ", 
                              command=self.select_voice_file, 
                              bg="red", fg="white", font=("Arial", 16, "bold"),
                              relief=tk.RAISED, bd=5, height=2, width=25)
        big_button.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Поле для отображения выбранного файла
        ttk.Label(file_frame, text="Выбранный файл:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        file_display_frame = ttk.Frame(file_frame)
        file_display_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        file_display_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(file_display_frame, textvariable=self.voice_file_path, state="readonly").grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # Дополнительная информация о поддерживаемых форматах
        ttk.Label(file_frame, text="Поддерживаемые форматы: WAV, MP3, FLAC, M4A", 
                 font=("Arial", 10), foreground="blue").grid(row=2, column=0, columnspan=3, 
                                                          sticky=tk.W, pady=(10, 0))
        
        # ДОПОЛНИТЕЛЬНАЯ КНОПКА ВЫБОРА ФАЙЛА - НАД ТЕКСТОМ!
        extra_file_button = tk.Button(main_frame, text="🔥 ЗАГРУЗИТЬ ФАЙЛ С ГОЛОСОМ 🔥", 
                                     command=self.select_voice_file, 
                                     bg="orange", fg="black", font=("Arial", 14, "bold"),
                                     relief=tk.RAISED, bd=4, height=2)
        extra_file_button.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Текст для озвучки
        ttk.Label(main_frame, text="Текст для озвучки:").grid(
            row=6, column=0, sticky=tk.W, pady=(20, 5))
        
        self.text_input = scrolledtext.ScrolledText(main_frame, height=6, width=80)
        self.text_input.grid(row=6, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                             pady=5)
        
        # Пример текста с ударениями
        example_text = """Привет! Это пример текста для озвучки вашим клонированным голосом.

🎯 ПРИМЕРЫ УПРАВЛЕНИЯ УДАРЕНИЯМИ:

1. Ударения через символ +:
   зам+ок (крепость)
   зам+ок (дверной механизм)
   компьют+ер
   интерн+ет

2. Эмфатические ударения:
   Это *ОЧЕНЬ* важно! 
   Это **важное** слово с двойным выделением.

3. Сложные слова с правильными ударениями:
   компьютер, интернет, телефон

4. Паузы и интонация:
   Это слово... с паузой.
   Это вопрос? 
   Это восклицание!

5. SSML разметка:
   <emphasis>Выделенное слово</emphasis>
   <break time="500ms"/> пауза

Программа использует XTTS v2 для клонирования голоса с поддержкой русского языка.
Вы можете вставить любой текст на русском языке, и он будет озвучен качественно.
Теперь у вас есть настоящий русский TTS с клонированием голоса!"""
        self.text_input.insert(tk.END, example_text)
        
        # Кнопки справки и тестирования
        help_button_frame = ttk.Frame(main_frame)
        help_button_frame.grid(row=7, column=0, columnspan=3, pady=5)
        
        ttk.Button(help_button_frame, text="🎯 Справка по ударениям", 
                   command=self.show_stress_help, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(help_button_frame, text="🧪 Тест ударений", 
                   command=self.test_stress_processing).pack(side=tk.LEFT)
        
        # Настройки
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки", padding="10")
        settings_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        settings_frame.columnconfigure(1, weight=1)
        
        # Скорость речи
        ttk.Label(settings_frame, text="Скорость речи:").grid(row=0, column=0, sticky=tk.W)
        self.speed_var = tk.DoubleVar(value=0.8)
        speed_scale = ttk.Scale(settings_frame, from_=0.2, to=1.0, 
                               variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        speed_label = ttk.Label(settings_frame, text="0.8")
        speed_label.grid(row=0, column=2)
        
        # Привязка обновления лейбла
        speed_scale.configure(command=lambda x: speed_label.configure(text=f"{float(x):.1f}"))
        
        # Кнопки генерации
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=9, column=0, columnspan=3, pady=20)
        
        self.process_button = ttk.Button(button_frame, text="🎯 XTTS v2 Клонировать голос", 
                                        command=self.process_text, style="Accent.TButton")
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.standard_button = ttk.Button(button_frame, text="🇷🇺 Windows TTS голос", 
                                         command=self.generate_windows_voice)
        self.standard_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Дублирующая кнопка выбора файла
        duplicate_file_button = ttk.Button(button_frame, text="📁 Выбрать файл", 
                                          command=self.select_voice_file, style="Accent.TButton")
        duplicate_file_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Кнопки воспроизведения
        playback_frame = ttk.LabelFrame(main_frame, text="Воспроизведение", padding="10")
        playback_frame.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        playback_frame.columnconfigure(1, weight=1)
        
        # XTTS v2 голос
        ttk.Label(playback_frame, text="XTTS v2 клонированный голос:").grid(row=0, column=0, sticky=tk.W, pady=5)
        cloned_buttons = ttk.Frame(playback_frame)
        cloned_buttons.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(cloned_buttons, text="▶️ Воспроизвести", 
                   command=self.play_cloned_audio).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(cloned_buttons, text="💾 Сохранить", 
                   command=self.save_cloned_audio).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(cloned_buttons, text="📊 Спектрограмма", 
                   command=self.show_cloned_spectrogram).pack(side=tk.LEFT)
        
        # Стандартный голос
        ttk.Label(playback_frame, text="Windows TTS голос:").grid(row=1, column=0, sticky=tk.W, pady=5)
        standard_buttons = ttk.Frame(playback_frame)
        standard_buttons.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(standard_buttons, text="▶️ Воспроизвести", 
                   command=self.play_standard_audio).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(standard_buttons, text="💾 Сохранить", 
                   command=self.save_standard_audio).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(standard_buttons, text="📊 Спектрограмма", 
                   command=self.show_standard_spectrogram).pack(side=tk.LEFT)
        
        # Прогресс
        self.progress_var = tk.StringVar(value="Готов к работе")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(
            row=11, column=0, columnspan=3, pady=10)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=12, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Статус моделей
        self.model_status = tk.StringVar(value="Статус моделей: Загрузка...")
        ttk.Label(main_frame, textvariable=self.model_status, 
                 font=("Arial", 9), foreground="green").grid(
            row=13, column=0, columnspan=3, pady=5)
        
        # Настройка весов для растягивания
        main_frame.rowconfigure(5, weight=1)
    
    def init_models(self):
        """Инициализация моделей TTS"""
        try:
            self.progress_var.set("Загрузка моделей...")
            self.root.update()
            
            # Пробуем загрузить XTTS v2 для клонирования
            self.progress_var.set("Загрузка XTTS v2 для клонирования...")
            self.root.update()
            
            # Исправление для PyTorch 2.6
            try:
                from torch.serialization import add_safe_globals
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import XttsAudioConfig
                from TTS.tts.models.xtts import XttsArgs
                from TTS.config.shared_configs import BaseDatasetConfig
                add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
                print("✅ PyTorch 2.6 safe globals добавлены в init_models (включая XttsArgs)")
            except Exception as e:
                print(f"⚠️ Ошибка добавления safe globals в init_models: {e}")
                pass
            
            try:
                # Возвращаемся к XTTS v2 - он поддерживает русский язык!
                self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                if self.xtts_model is None:
                    raise Exception("XTTS v2 не загрузился")
                
                # Проверяем поддерживаемые языки
                supported_langs = getattr(self.xtts_model, 'languages', [])
                print(f"Поддерживаемые языки XTTS v2: {supported_langs}")
                
                # XTTS v2 поддерживает русский язык!
                if 'ru' in supported_langs:
                    xtts_status = "✅ XTTS v2 загружена - РУССКИЙ ПОДДЕРЖИВАЕТСЯ!"
                else:
                    xtts_status = "⚠️ XTTS v2 загружена - проверяем поддержку русского..."
            except Exception as e:
                print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: XTTS v2 не загружается: {e}")
                self.xtts_model = None
                xtts_status = "❌ XTTS v2 НЕ ЗАГРУЖЕНА - русское клонирование невозможно!"
                # Показываем критическое сообщение
                error_msg = str(e)
                self.root.after(0, lambda: messagebox.showerror(
                    "Критическая ошибка", 
                    f"Не удалось загрузить XTTS v2 для русского клонирования!\n\n"
                    f"Ошибка: {error_msg}\n\n"
                    f"Без XTTS v2 невозможно качественное клонирование голоса на русском языке.\n"
                    f"Проверьте интернет-соединение и попробуйте перезапустить программу."
                ))
            
            # Инициализируем системный TTS Windows
            self.progress_var.set("Инициализация Windows TTS...")
            self.root.update()
            try:
                self.windows_tts = pyttsx3.init()
                
                # Получаем доступные голоса
                voices = self.windows_tts.getProperty('voices')
                russian_voice = None
                
                # Ищем русский голос
                for voice in voices:
                    if 'russian' in voice.name.lower() or 'ru' in voice.id.lower():
                        russian_voice = voice
                        break
                
                if russian_voice:
                    self.windows_tts.setProperty('voice', russian_voice.id)
                    windows_status = f"✅ Windows TTS с русским голосом: {russian_voice.name}"
                else:
                    # Используем первый доступный голос
                    if voices:
                        self.windows_tts.setProperty('voice', voices[0].id)
                        windows_status = f"✅ Windows TTS: {voices[0].name}"
                    else:
                        windows_status = "⚠️ Windows TTS: голоса не найдены"
                        
            except Exception as e:
                print(f"Ошибка Windows TTS: {e}")
                windows_status = "❌ Windows TTS: ошибка инициализации"
            
            self.model_status.set(f"Статус: {xtts_status} | {windows_status}")
            self.progress_var.set("Готов к работе")
            
        except Exception as e:
            error_msg = f"Не удалось загрузить модели: {str(e)}"
            messagebox.showerror("Ошибка", error_msg)
            self.progress_var.set("Ошибка загрузки моделей")
            self.model_status.set("Статус моделей: ❌ Ошибка загрузки")
    
    def start_recording(self):
        """Начать запись через микрофон"""
        if self.is_recording:
            return
        
        try:
            # Создание временного файла для записи
            self.recorded_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            self.recorded_file_path = self.recorded_file.name
            self.recorded_file.close()
            
            # Настройка записи
            self.frames = []
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            self.is_recording = True
            self.record_start_time = time.time()
            
            # Обновление интерфейса
            self.record_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.recording_indicator.config(text="🔴", foreground="red")
            
            # Запуск записи в отдельном потоке
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            # Запуск таймера
            self.update_timer()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось начать запись: {str(e)}")
    
    def _record_audio(self):
        """Запись аудио в отдельном потоке"""
        try:
            while self.is_recording:
                data = self.stream.read(self.CHUNK)
                self.frames.append(data)
        except Exception as e:
            print(f"Ошибка записи: {e}")
        finally:
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
    
    def update_timer(self):
        """Обновление таймера записи"""
        if self.is_recording:
            elapsed = int(time.time() - self.record_start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.recording_timer.set(f"{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)
    
    def stop_recording(self):
        """Остановить запись"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Ожидание завершения записи
        if self.recording_thread:
            self.recording_thread.join()
        
        # Сохранение записи в файл
        try:
            with wave.open(self.recorded_file_path, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.frames))
            
            # Установка пути к записанному файлу
            self.voice_file_path.set(self.recorded_file_path)
            
            # Проверка длительности
            duration = len(self.frames) * self.CHUNK / self.RATE
            if duration < 5:
                messagebox.showwarning("Предупреждение", 
                                     f"Запись слишком короткая ({duration:.1f} сек). "
                                     "Рекомендуется минимум 30 секунд.")
            elif duration > 300:
                messagebox.showwarning("Предупреждение", 
                                     f"Запись очень длинная ({duration:.1f} сек). "
                                     "Рекомендуется 1-2 минуты.")
            else:
                messagebox.showinfo("Успех", 
                                   f"Запись завершена ({duration:.1f} сек). "
                                   "Можно начинать клонирование!")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить запись: {str(e)}")
        
        # Обновление интерфейса
        self.record_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.recording_indicator.config(text="", foreground="black")
        self.recording_timer.set("00:00")
    
    def select_voice_file(self):
        """Выбор файла с голосом"""
        file_path = filedialog.askopenfilename(
            title="Выберите аудиофайл с вашим голосом",
            filetypes=[
                ("Аудио файлы", "*.wav *.mp3 *.flac *.m4a"),
                ("Все файлы", "*.*")
            ]
        )
        if file_path:
            self.voice_file_path.set(file_path)
            
            # Проверка длительности файла
            try:
                y, sr = librosa.load(file_path)
                duration = len(y) / sr
                if duration < 10:
                    messagebox.showwarning("Предупреждение", 
                                         f"Файл слишком короткий ({duration:.1f} сек). "
                                         "Рекомендуется минимум 30 секунд для лучшего качества.")
                elif duration > 300:
                    messagebox.showwarning("Предупреждение", 
                                         f"Файл очень длинный ({duration:.1f} сек). "
                                         "Рекомендуется 1-2 минуты для оптимального качества.")
                else:
                    messagebox.showinfo("Информация", 
                                       f"Файл подходящей длины ({duration:.1f} сек). "
                                       "Можно начинать клонирование!")
            except Exception as e:
                print(f"Ошибка при анализе файла: {e}")
    
    def process_text(self):
        """Обработка текста и создание аудио с клонированием голоса через XTTS v2"""
        if self.is_processing:
            return
        
        if not self.voice_file_path.get():
            messagebox.showwarning("Предупреждение", "Сначала запишите или выберите файл с вашим голосом!")
            return
        
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Предупреждение", "Введите текст для озвучки!")
            return
        
        # Запуск обработки в отдельном потоке
        self.is_processing = True
        self.process_button.config(state="disabled")
        self.progress_bar.start()
        
        thread = threading.Thread(target=self._process_text_thread, args=(text,))
        thread.daemon = True
        thread.start()
    
    def process_text_with_stress(self, text):
        """Обработка текста с учетом ударений и специальных символов"""
        import re
        
        # Обработка ударений через символ + (работает в обе стороны: +а и а+)
        def process_plus_stress(match):
            word = match.group(0)
            # Находим позицию + в слове
            plus_pos = word.find('+')
            if plus_pos != -1:
                # Убираем + и определяем ударную букву
                clean_word = word.replace('+', '')
                
                # Определяем ударную букву (перед или после +)
                if plus_pos > 0 and plus_pos < len(clean_word):
                    # + между буквами - берем букву после +
                    stressed_letter = clean_word[plus_pos]
                    stressed_pos = plus_pos
                elif plus_pos > 0:
                    # + в конце - берем букву перед +
                    stressed_letter = clean_word[plus_pos-1]
                    stressed_pos = plus_pos-1
                else:
                    # + в начале - берем букву после +
                    stressed_letter = clean_word[0]
                    stressed_pos = 0
                
                # Проверяем, что это гласная, если нет - ищем ближайшую гласную
                vowels = 'аеёиоуыэюя'
                if stressed_letter.lower() not in vowels:
                    # Ищем ближайшую гласную к позиции + (не перескакиваем через согласные)
                    # Сначала ищем слева от + (включая позицию +)
                    left_search = range(stressed_pos, -1, -1)
                    # Потом справа от + (если слева не нашли)
                    right_search = range(stressed_pos + 1, len(clean_word))
                    
                    # Объединяем поиски
                    for pos in list(left_search) + list(right_search):
                        if pos < len(clean_word) and clean_word[pos].lower() in vowels:
                            stressed_letter = clean_word[pos]
                            stressed_pos = pos
                            break
                    else:
                        # Если гласную не нашли, возвращаем слово без изменений
                        print(f"Не найдена гласная в слове '{word}'")
                        return clean_word
                
                # Заменяем на готовые буквы с ударениями (один символ)
                stressed_vowels = {
                    'а': 'á',  # а с острым ударением (один символ)
                    'о': 'ó',  # о с острым ударением (один символ)
                    'е': 'é',  # е с острым ударением (один символ)
                    'и': 'í',  # и с острым ударением (один символ)
                    'у': 'ú',  # у с острым ударением (один символ)
                    'ы': 'ы́',  # ы с острым ударением (два символа, но это стандарт)
                    'э': 'э́',  # э с острым ударением (два символа, но это стандарт)
                    'ю': 'ю́',  # ю с острым ударением (два символа, но это стандарт)
                    'я': 'я́',  # я с острым ударением (два символа, но это стандарт)
                    'ё': 'ё́',  # ё с острым ударением (два символа, но это стандарт)
                }
                
                # Заменяем гласную на версию с ударением
                stressed_letter_new = stressed_vowels.get(stressed_letter.lower(), stressed_letter)
                # Сохраняем регистр
                if stressed_letter.isupper():
                    stressed_letter_new = stressed_letter_new.upper()
                
                stressed_word = clean_word[:stressed_pos] + stressed_letter_new + clean_word[stressed_pos+1:]
                print(f"Обработка ударения: '{word}' -> '{stressed_word}' (ударение на букву '{stressed_letter}' -> '{stressed_letter_new}')")
                return stressed_word
            return word
        
        # Обрабатываем слова с + ударениями (обрабатываем каждый + отдельно)
        def process_word_with_pluses(word):
            try:
                # Обрабатываем каждый + в слове отдельно
                while '+' in word:
                    # Находим позицию первого +
                    plus_pos = word.find('+')
                    if plus_pos == -1:
                        break
                    
                    # Проверяем границы строки перед обращением к индексам
                    if len(word) <= 1:
                        print(f"Слово слишком короткое для обработки ударения: '{word}'")
                        break
                    
                    # Определяем ударную букву (перед или после +)
                    if plus_pos > 0 and plus_pos < len(word) - 1:
                        # + между буквами - берем букву после +
                        stressed_letter = word[plus_pos + 1]
                        stressed_pos = plus_pos + 1
                    elif plus_pos > 0:
                        # + в конце - берем букву перед +
                        stressed_letter = word[plus_pos - 1]
                        stressed_pos = plus_pos - 1
                    else:
                        # + в начале - берем букву после +
                        if len(word) > 1:
                            stressed_letter = word[1]
                            stressed_pos = 1
                        else:
                            print(f"Слово слишком короткое для обработки ударения: '{word}'")
                            word = word.replace('+', '', 1)
                            continue
                    
                    # Проверяем, что это гласная, если нет - ищем ближайшую гласную
                    vowels = 'аеёиоуыэюя'
                    if stressed_letter.lower() not in vowels:
                        # Ищем ближайшую гласную к позиции + (не перескакиваем через согласные)
                        # Сначала ищем слева от + (включая позицию +)
                        left_search = range(stressed_pos, -1, -1)
                        # Потом справа от + (если слева не нашли)
                        right_search = range(stressed_pos + 1, len(word))
                        
                        # Объединяем поиски
                        for pos in list(left_search) + list(right_search):
                            if pos < len(word) and word[pos].lower() in vowels:
                                stressed_letter = word[pos]
                                stressed_pos = pos
                                break
                        else:
                            # Если гласную не нашли, просто убираем +
                            word = word.replace('+', '', 1)
                            continue
                    
                    # Проверяем, что гласная еще не ударена
                    stressed_vowels = 'áóéíúы́э́ю́я́ё́'
                    if stressed_letter in stressed_vowels:
                        # Гласная уже ударена, просто убираем +
                        word = word.replace('+', '', 1)
                        print(f"Гласная '{stressed_letter}' уже ударена, + игнорируется")
                        continue
                    
                    # Заменяем на готовые буквы с ударениями (один символ)
                    stressed_vowels_dict = {
                        'а': 'á',  # а с острым ударением (один символ)
                        'о': 'ó',  # о с острым ударением (один символ)
                        'е': 'é',  # е с острым ударением (один символ)
                        'и': 'í',  # и с острым ударением (один символ)
                        'у': 'ú',  # у с острым ударением (один символ)
                        'ы': 'ы́',  # ы с острым ударением (два символа, но это стандарт)
                        'э': 'э́',  # э с острым ударением (два символа, но это стандарт)
                        'ю': 'ю́',  # ю с острым ударением (два символа, но это стандарт)
                        'я': 'я́',  # я с острым ударением (два символа, но это стандарт)
                        'ё': 'ё́',  # ё с острым ударением (два символа, но это стандарт)
                    }
                    
                    # Заменяем гласную на версию с ударением
                    stressed_letter_new = stressed_vowels_dict.get(stressed_letter.lower(), stressed_letter)
                    # Сохраняем регистр
                    if stressed_letter.isupper():
                        stressed_letter_new = stressed_letter_new.upper()
                    
                    # Заменяем букву и убираем +
                    word = word[:stressed_pos] + stressed_letter_new + word[stressed_pos + 1:]
                    word = word.replace('+', '', 1)  # Убираем только первый +
                    
                    print(f"Обработка ударения: ударение на букву '{stressed_letter}' -> '{stressed_letter_new}'")
                
                return word
            except Exception as e:
                print(f"Ошибка при обработке ударений в слове '{word}': {e}")
                # В случае ошибки просто убираем все + из слова
                return word.replace('+', '')
        
        # Обрабатываем каждое слово с + отдельно
        def process_plus_stress_in_word(match):
            word = match.group(0)
            return process_word_with_pluses(word)
        
        text = re.sub(r'\b[а-яёА-ЯЁ]*\+[а-яёА-ЯЁ]*\b', process_plus_stress_in_word, text)
        
        # Метод ослабления ударения (используем символ -) - работает в обе стороны: -а и а-
        def process_weak_stress(match):
            word = match.group(0)
            try:
                if '-' in word:
                    # Находим позицию - в слове
                    minus_pos = word.find('-')
                    # Убираем - и определяем ослабляемую букву
                    clean_word = word.replace('-', '')
                    
                    # Проверяем границы строки перед обращением к индексам
                    if len(clean_word) == 0:
                        print(f"Слово пустое после удаления -: '{word}'")
                        return word
                    
                    # Определяем ослабляемую букву (перед или после -)
                    if minus_pos > 0 and minus_pos < len(clean_word):
                        # - между буквами - берем букву после -
                        weak_letter = clean_word[minus_pos]
                        weak_pos = minus_pos
                    elif minus_pos > 0:
                        # - в конце - берем букву перед -
                        weak_letter = clean_word[minus_pos-1]
                        weak_pos = minus_pos-1
                    else:
                        # - в начале - берем букву после -
                        weak_letter = clean_word[0]
                        weak_pos = 0
                    
                    # Проверяем, что это гласная, если нет - ищем ближайшую гласную
                    vowels = 'аеёиоуыэюя'
                    if weak_letter.lower() not in vowels:
                        # Ищем ближайшую гласную к позиции - (не перескакиваем через согласные)
                        # Сначала ищем слева от - (включая позицию -)
                        left_search = range(weak_pos, -1, -1)
                        # Потом справа от - (если слева не нашли)
                        right_search = range(weak_pos + 1, len(clean_word))
                        
                        # Объединяем поиски
                        for pos in list(left_search) + list(right_search):
                            if pos < len(clean_word) and clean_word[pos].lower() in vowels:
                                weak_letter = clean_word[pos]
                                weak_pos = pos
                                break
                        else:
                            # Если гласную не нашли, возвращаем слово без изменений
                            print(f"Не найдена гласная в слове '{word}'")
                            return clean_word
                    
                    # Заменяем на более слабую гласную
                    weak_vowels = {
                        'а': 'ə',  # schwa - нейтральная гласная
                        'о': 'ə',  # schwa
                        'е': 'ɪ',  # короткая i
                        'и': 'ɪ',  # короткая i
                        'у': 'ʊ',  # короткая u
                        'ы': 'ə',  # schwa
                        'э': 'ɛ',  # короткая e
                        'ю': 'ʊ',  # короткая u
                        'я': 'ə',  # schwa
                        'ё': 'ɪ',  # короткая i
                    }
                    
                    # Заменяем гласную на более слабую
                    weak_letter_new = weak_vowels.get(weak_letter.lower(), weak_letter)
                    # Сохраняем регистр
                    if weak_letter.isupper():
                        weak_letter_new = weak_letter_new.upper()
                    
                    weak_word = clean_word[:weak_pos] + weak_letter_new + clean_word[weak_pos+1:]
                    print(f"Ослабление ударения: '{word}' -> '{weak_word}' (ослаблена буква '{weak_letter}' -> '{weak_letter_new}')")
                    return weak_word
                return word
            except Exception as e:
                print(f"Ошибка при обработке ослабления ударений в слове '{word}': {e}")
                # В случае ошибки просто убираем все - из слова
                return word.replace('-', '')
        
        # Применяем метод ослабления ударения (слова с символом -)
        text = re.sub(r'\b[а-яёА-ЯЁ]*-[а-яёА-ЯЁ]*\b', process_weak_stress, text)
        
        # Словарь правильных ударений для сложных случаев
        stress_dict = {
            "компьютер": "компьютер",
            "интернет": "интернет", 
            "телефон": "телефон",
            "одновременно": "одновременно",
            "одновременно": "одновременно",
            "замок": "замок",  # крепость (з+амок)
            "замок": "замок",  # дверной механизм (зам+ок)
            "мука": "мука",    # страдание (м+ука)
            "мука": "мука",    # продукт (мук+а)
            "Федотов": "Федотов",  # Фед+отов
            "века": "века",    # в+ека
        }
        
        # Обработка SSML тегов
        text = re.sub(r'<emphasis>(.*?)</emphasis>', r'*\1*', text)
        text = re.sub(r'<break time="(\d+)ms"/>', r'...', text)
        
        # Обработка эмфатических ударений
        text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)  # Двойные звездочки
        text = re.sub(r'__(.*?)__', r'*\1*', text)      # Подчеркивание
        
        # Обработка пауз
        text = re.sub(r'\.{3,}', '...', text)  # Множественные точки
        
        # Замена сложных слов с правильными ударениями
        for word, stressed_word in stress_dict.items():
            text = re.sub(r'\b' + re.escape(word) + r'\b', stressed_word, text, flags=re.IGNORECASE)
        
        return text

    def split_text_for_xtts(self, text, max_length=150):
        """Разбивает текст на части подходящие для XTTS v2"""
        import re
        
        # Разбиваем на предложения (по точкам, восклицательным, вопросительным знакам и переносам строк)
        sentences = re.split(r'[.!?\n]+', text)
        chunks = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение короткое, добавляем как есть
            if len(sentence) <= max_length:
                chunks.append(sentence)
                continue
            
            # Если предложение длинное, разбиваем по приоритету разделителей
            chunks.extend(self._split_long_sentence(sentence, max_length))
        
        return chunks
    
    def _split_long_sentence(self, sentence, max_length):
        """Разбивает длинное предложение по приоритету разделителей"""
        import re
        
        # Приоритет разделителей: запятые и точки с запятой -> тире -> двоеточие -> пробелы
        separators = [
            r'[,;]',      # Запятые и точки с запятой
            r'\s+-\s+',   # Тире с пробелами
            r':',         # Двоеточие
            r'\s+'        # Любые пробелы (последний случай)
        ]
        
        for i, separator in enumerate(separators):
            parts = re.split(separator, sentence)
            
            # Если разбиение дало больше одной части
            if len(parts) > 1:
                result_chunks = []
                current_part = ""
                
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    # Если часть сама по себе слишком длинная
                    if len(part) > max_length:
                        # Сначала добавляем накопленную часть
                        if current_part:
                            result_chunks.append(current_part.strip())
                            current_part = ""
                        
                        # Если это последний разделитель (пробелы), разбиваем на слова
                        if i == len(separators) - 1:
                            words = part.split()
                            temp_chunk = ""
                            
                            for word in words:
                                if len(temp_chunk + " " + word) <= max_length:
                                    temp_chunk += (" " + word) if temp_chunk else word
                                else:
                                    if temp_chunk:
                                        result_chunks.append(temp_chunk.strip())
                                    temp_chunk = word
                            
                            if temp_chunk:
                                current_part = temp_chunk
                        else:
                            # Пробуем следующий разделитель для этой части
                            sub_chunks = self._split_long_sentence(part, max_length)
                            if len(sub_chunks) > 1:
                                # Если подразбиение удалось, добавляем все части
                                if current_part:
                                    result_chunks.append(current_part.strip())
                                    current_part = ""
                                result_chunks.extend(sub_chunks)
                            else:
                                # Если подразбиение не удалось, добавляем как есть
                                if current_part:
                                    result_chunks.append(current_part.strip())
                                    current_part = ""
                                result_chunks.append(part)
                    else:
                        # Проверяем, поместится ли часть в текущий чанк
                        if len(current_part + " " + part) <= max_length:
                            current_part += (" " + part) if current_part else part
                        else:
                            if current_part:
                                result_chunks.append(current_part.strip())
                            current_part = part
                
                # Добавляем последнюю часть
                if current_part:
                    result_chunks.append(current_part.strip())
                
                # Если получилось разбиение, возвращаем результат
                if len(result_chunks) > 1:
                    return result_chunks
        
        # Если ни один разделитель не помог, возвращаем предложение как есть
        return [sentence]

    def _process_text_thread(self, text):
        """Поток для обработки текста с клонированием через XTTS v2"""
        try:
            # Проверяем, загружена ли модель
            if self.xtts_model is None:
                raise Exception("Модель клонирования не загружена. Попробуйте перезапустить программу.")
            
            self.root.after(0, lambda: self.progress_var.set("Анализ вашего голоса..."))
            
            # Обработка ударений в тексте
            processed_text = self.process_text_with_stress(text)
            print(f"📝 Текст обработан с учетом ударений")
            print(f"📝 Исходный текст: {text[:100]}...")
            print(f"📝 Обработанный текст: {processed_text[:100]}...")
            
            # Разбиваем текст на части если он слишком длинный
            text_chunks = self.split_text_for_xtts(processed_text)
            print(f"📝 Текст разбит на {len(text_chunks)} частей для обработки")
            
            if len(text_chunks) > 1:
                self.root.after(0, lambda: self.progress_var.set(f"Обработка длинного текста ({len(text_chunks)} частей)..."))
            
            # Создание одного WAV файла для последовательной записи
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as final_file:
                output_path = final_file.name
            
            import soundfile as sf
            import numpy as np
            
            # Список для хранения всех аудио данных
            all_audio_data = []
            sample_rate = None
            
            # Генерируем каждую часть и добавляем в общий массив
            for i, chunk in enumerate(text_chunks):
                if len(text_chunks) > 1:
                    self.root.after(0, lambda i=i, total=len(text_chunks): 
                        self.progress_var.set(f"Генерация части {i+1} из {total}..."))
                
                # Создание временного файла для части
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    chunk_output_path = tmp_file.name
                
                # XTTS v2 с исправленными параметрами
                try:
                    self.xtts_model.tts_to_file(
                        text=chunk,
                        speaker_wav=self.voice_file_path.get(),
                        file_path=chunk_output_path,
                        language="ru",
                        # Только поддерживаемые параметры
                        speed=1.0,
                        temperature=0.7,
                        length_penalty=1.0,
                        repetition_penalty=2.0,
                        top_k=50,
                        top_p=0.85
                    )
                    
                    # Читаем сгенерированную часть и добавляем в общий массив
                    audio_data, sr = sf.read(chunk_output_path)
                    if sample_rate is None:
                        sample_rate = sr
                    all_audio_data.append(audio_data)
                    
                    # Удаляем временный файл части
                    try:
                        os.unlink(chunk_output_path)
                    except:
                        pass
                    
                    print(f"✅ Часть {i+1} сгенерирована и добавлена в общий файл")
                    
                except Exception as e:
                    print(f"Ошибка с частью {i+1}: {e}")
                    # Пробуем с минимальными параметрами
                    try:
                        self.xtts_model.tts_to_file(
                            text=chunk,
                            speaker_wav=self.voice_file_path.get(),
                            file_path=chunk_output_path,
                            language="ru"
                        )
                        
                        # Читаем сгенерированную часть и добавляем в общий массив
                        audio_data, sr = sf.read(chunk_output_path)
                        if sample_rate is None:
                            sample_rate = sr
                        all_audio_data.append(audio_data)
                        
                        # Удаляем временный файл части
                        try:
                            os.unlink(chunk_output_path)
                        except:
                            pass
                        
                        print(f"✅ Часть {i+1} сгенерирована с базовыми параметрами")
                    except Exception as e2:
                        raise Exception(f"Не удалось сгенерировать часть {i+1}: {e2}")
            
            # Объединяем все части в один файл
            if len(all_audio_data) > 1:
                self.root.after(0, lambda: self.progress_var.set("Сохранение объединенного аудио..."))
                
                # Объединяем все аудио данные
                final_audio = np.concatenate(all_audio_data)
                sf.write(output_path, final_audio, sample_rate)
                
                print(f"✅ Объединено {len(all_audio_data)} частей в один файл")
                
            else:
                # Если только одна часть, сохраняем её как есть
                sf.write(output_path, all_audio_data[0], sample_rate)
                print("✅ Аудио сохранено в один файл")
            
            # Сохранение пути к результату
            self.output_path.set(output_path)
            
            self.root.after(0, lambda: self.progress_var.set("Клонирование голоса завершено успешно!"))
            
        except Exception as e:
            error_msg = f"Не удалось клонировать голос: {str(e)}"
            self.root.after(0, lambda: self.progress_var.set(f"Ошибка: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))
        
        finally:
            self.root.after(0, lambda: self._finish_processing())
    
    def _finish_processing(self):
        """Завершение обработки в главном потоке"""
        self.is_processing = False
        self.progress_bar.stop()
        self.process_button.config(state="normal")
    
    def generate_windows_voice(self):
        """Генерация Windows TTS голоса"""
        if self.is_processing:
            return
        
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Предупреждение", "Введите текст для озвучки!")
            return
        
        # Запуск обработки в отдельном потоке
        self.is_processing = True
        self.standard_button.config(state="disabled")
        self.progress_bar.start()
        
        thread = threading.Thread(target=self._generate_windows_thread, args=(text,))
        thread.daemon = True
        thread.start()
    
    def _generate_windows_thread(self, text):
        """Поток для генерации Windows TTS голоса"""
        try:
            self.root.after(0, lambda: self.progress_var.set("Генерация Windows TTS голоса..."))
            
            # Создание временного файла для результата
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # Настройка Windows TTS
            self.windows_tts.setProperty('rate', int(150 * self.speed_var.get()))
            
            # Сохранение в файл
            self.windows_tts.save_to_file(text, output_path)
            self.windows_tts.runAndWait()
            
            # Сохранение пути к результату
            self.standard_output_path.set(output_path)
            
            self.root.after(0, lambda: self.progress_var.set("Windows TTS голос сгенерирован успешно!"))
            
        except Exception as e:
            error_msg = f"Не удалось сгенерировать Windows TTS голос: {str(e)}"
            self.root.after(0, lambda: self.progress_var.set(f"Ошибка: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))
        
        finally:
            self.root.after(0, lambda: self._finish_windows_processing())
    
    def _finish_windows_processing(self):
        """Завершение обработки Windows TTS в главном потоке"""
        self.is_processing = False
        self.progress_bar.stop()
        self.standard_button.config(state="normal")
    
    def play_cloned_audio(self):
        """Воспроизведение клонированного аудио"""
        if not self.output_path.get():
            messagebox.showwarning("Предупреждение", "Сначала клонируйте голос!")
            return
        
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                os.startfile(self.output_path.get())
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", self.output_path.get()])
            else:  # Linux
                subprocess.run(["xdg-open", self.output_path.get()])
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось воспроизвести аудио: {str(e)}")
    
    def play_standard_audio(self):
        """Воспроизведение стандартного аудио"""
        if not self.standard_output_path.get():
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте Windows TTS голос!")
            return
        
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                os.startfile(self.standard_output_path.get())
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", self.standard_output_path.get()])
            else:  # Linux
                subprocess.run(["xdg-open", self.standard_output_path.get()])
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось воспроизвести аудио: {str(e)}")
    
    def save_cloned_audio(self):
        """Сохранение клонированного аудио файла"""
        if not self.output_path.get():
            messagebox.showwarning("Предупреждение", "Сначала клонируйте голос!")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Сохранить XTTS v2 клонированный аудио файл",
            defaultextension=".wav",
            filetypes=[
                ("WAV файлы", "*.wav"),
                ("MP3 файлы", "*.mp3"),
                ("Все файлы", "*.*")
            ]
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy2(self.output_path.get(), save_path)
                messagebox.showinfo("Успех", f"XTTS v2 клонированный аудио сохранен в: {save_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")
    
    def save_standard_audio(self):
        """Сохранение стандартного аудио файла"""
        if not self.standard_output_path.get():
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте Windows TTS голос!")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Сохранить Windows TTS аудио файл",
            defaultextension=".wav",
            filetypes=[
                ("WAV файлы", "*.wav"),
                ("MP3 файлы", "*.mp3"),
                ("Все файлы", "*.*")
            ]
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy2(self.standard_output_path.get(), save_path)
                messagebox.showinfo("Успех", f"Windows TTS аудио сохранен в: {save_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")
    
    def show_cloned_spectrogram(self):
        """Показать спектрограмму клонированного аудио"""
        if not self.output_path.get():
            messagebox.showwarning("Предупреждение", "Сначала клонируйте голос!")
            return
        
        self._show_spectrogram(self.output_path.get(), "XTTS v2 клонированный голос")
    
    def show_standard_spectrogram(self):
        """Показать спектрограмму стандартного аудио"""
        if not self.standard_output_path.get():
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте Windows TTS голос!")
            return
        
        self._show_spectrogram(self.standard_output_path.get(), "Windows TTS голос")
    
    def show_stress_help(self):
        """Показать справку по ударениям в отдельном окне"""
        help_window = tk.Toplevel(self.root)
        help_window.title("🎯 Справка по управлению ударениями")
        help_window.geometry("600x500")
        help_window.resizable(True, True)
        
        # Создание фрейма с прокруткой
        main_frame = ttk.Frame(help_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="📝 СИМВОЛЫ ДЛЯ УПРАВЛЕНИЯ УДАРЕНИЯМИ", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Создание текстового виджета с прокруткой
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                               font=("Arial", 11), 
                                               bg="white", fg="black")
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Содержимое справки
        help_text = """🎯 СИМВОЛЫ ДЛЯ УПРАВЛЕНИЯ УДАРЕНИЯМИ В XTTS v2

1. УСИЛЕНИЕ УДАРЕНИЯ (символ +):
   • з+амок - ударение на первом слоге (крепость)
   • зам+ок - ударение на втором слоге (дверной механизм)
   • компьют+ер - ударение на втором слоге
   • интерн+ет - ударение на третьем слоге
   • телеф+он - ударение на втором слоге
   • одноврем+енно - ударение на четвертом слоге
   • Фед+отов - ударение на первом слоге
   • в+ека - ударение на первом слоге

2. ОСЛАБЛЕНИЕ УДАРЕНИЯ (символ -):
   • компьют-ер - ослабленная "е" (ə)
   • интерн-ет - ослабленная "е" (ə)
   • телеф-он - ослабленная "о" (ə)
   • машин-а - ослабленная "а" (ə)

2. ЭМФАТИЧЕСКИЕ УДАРЕНИЯ:
   • *слово* - эмфатическое ударение
   • **слово** - двойное выделение
   • __слово__ - подчеркивание

3. ПАУЗЫ И ИНТОНАЦИЯ:
   • ... - пауза (многоточие)
   • ? - вопросительная интонация
   • ! - восклицательная интонация

4. SSML РАЗМЕТКА:
   • <emphasis>слово</emphasis> - SSML выделение
   • <break time="500ms"/> - пауза в миллисекундах
   • <prosody rate="slow">медленная речь</prosody>
   • <prosody pitch="high">высокий тон</prosody>

💡 ПРАКТИЧЕСКИЕ ПРИМЕРЫ:

• "Это *ОЧЕНЬ* важно!" → выделение слова
• "з+амок на горе" → крепость (ударение на "за")
• "дверной зам+ок" → механизм (ударение на "мо")
• "компьют+ер работает быстро" → ударение на "те"
• "Фед+отов" → ударение на "Фед" (не "Фёд")
• "в+ека" → ударение на "ве" (не "вье")
• "Это слово... с паузой" → пауза
• "Что происходит?" → вопросительная интонация
• "Отлично!" → восклицательная интонация

🔍 КАК РАБОТАЕТ УДАРЕНИЕ:
• Символ + показывает, где должно быть ударение
• Программа заменяет + на специальный символ ударения Unicode
• XTTS v2 читает символ ударения и ставит ударение на нужный слог

🔧 КОМБИНИРОВАНИЕ МЕТОДОВ:
• "Это *компьют+ер* очень быстрый!" → ударение + эмфаза
• "зам+ок... на горе" → ударение + пауза
• "Что происходит с телеф+оном?" → ударение + вопрос

📋 РЕКОМЕНДАЦИИ:
• Используйте символ + для точного указания ударения
• Не переусердствуйте с выделениями
• Тестируйте результат прослушиванием
• Комбинируйте методы для лучшего эффекта"""
        
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # Делаем только для чтения
        
        # Кнопка закрытия
        close_button = ttk.Button(main_frame, text="Закрыть", 
                                 command=help_window.destroy)
        close_button.pack(pady=(20, 0))

    def test_stress_processing(self):
        """Тестирование обработки ударений"""
        test_text = """Тест обработки ударений:

УСИЛЕНИЕ УДАРЕНИЯ (символ +) - работает в обе стороны:
з+амок (крепость) -> за́мок
зам+ок (дверной механизм) -> замо́к
+замок (альтернативный способ) -> за́мок
компьют+ер -> компьюте́р
+компьютер (альтернативный способ) -> ко́мпьютер
интерн+ет -> интерне́т
телеф+он -> телефо́н
одноврем+енно -> одновреме́нно
Фед+отов -> Федо́тов
в+ека -> ве́ка

ОСЛАБЛЕНИЕ УДАРЕНИЯ (символ -) - работает в обе стороны:
компьют-ер (ослабленная "е") -> компьютəр
-компьютер (альтернативный способ) -> кəмпьютер
интерн-ет (ослабленная "е") -> интернət
телеф-он (ослабленная "о") -> телефən

ПРИМЕРЫ:
Я Сергей Фед+отов, я использую эти знания почти четверть в+ека.
Это *ОЧЕНЬ* важно!
Это слово... с паузой."""
        
        processed_text = self.process_text_with_stress(test_text)
        
        # Показываем результат в отдельном окне
        test_window = tk.Toplevel(self.root)
        test_window.title("🧪 Тест обработки ударений")
        test_window.geometry("700x500")
        
        # Создание фрейма
        main_frame = ttk.Frame(test_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="🧪 РЕЗУЛЬТАТ ОБРАБОТКИ УДАРЕНИЙ", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Исходный текст
        ttk.Label(main_frame, text="ИСХОДНЫЙ ТЕКСТ:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        original_text = scrolledtext.ScrolledText(main_frame, height=8, width=80, 
                                                font=("Arial", 10), bg="lightgray")
        original_text.pack(fill=tk.X, pady=(5, 15))
        original_text.insert(tk.END, test_text)
        original_text.config(state=tk.DISABLED)
        
        # Обработанный текст
        ttk.Label(main_frame, text="ОБРАБОТАННЫЙ ТЕКСТ:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        processed_text_widget = scrolledtext.ScrolledText(main_frame, height=8, width=80, 
                                                        font=("Arial", 10), bg="lightgreen")
        processed_text_widget.pack(fill=tk.X, pady=(5, 15))
        processed_text_widget.insert(tk.END, processed_text)
        processed_text_widget.config(state=tk.DISABLED)
        
        # Кнопка закрытия
        close_button = ttk.Button(main_frame, text="Закрыть", 
                                 command=test_window.destroy)
        close_button.pack(pady=(10, 0))

    def _show_spectrogram(self, audio_path, title):
        """Показать спектрограмму аудио"""
        try:
            # Загрузка аудио
            y, sr = librosa.load(audio_path)
            
            # Создание спектрограммы
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            # Создание окна с графиком
            spectrogram_window = tk.Toplevel(self.root)
            spectrogram_window.title(f"Спектрограмма - {title}")
            spectrogram_window.geometry("900x600")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Спектрограмма
            img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax1)
            ax1.set_title(f'Спектрограмма - {title}')
            fig.colorbar(img, ax=ax1, format="%+2.f dB")
            
            # Волновая форма
            librosa.display.waveshow(y, sr=sr, ax=ax2)
            ax2.set_title('Волновая форма')
            ax2.set_xlabel('Время (секунды)')
            ax2.set_ylabel('Амплитуда')
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, spectrogram_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать спектрограмму: {str(e)}")
    
    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, 'audio') and self.audio is not None:
            try:
                self.audio.terminate()
            except Exception as e:
                print(f"⚠️ Ошибка при закрытии PyAudio: {e}")

def main():
    """Главная функция приложения с обработкой ошибок дисплея"""
    try:
        print("🚀 Запуск приложения клонирования голоса...")
        root = tk.Tk()
        app = VoiceClonerXTTSApp(root)
        root.mainloop()
    except Exception as e:
        print(f"❌ Ошибка запуска приложения: {e}")
        print("💡 Возможные решения:")
        print("   1. Установите pyvirtualdisplay: pip install pyvirtualdisplay")
        print("   2. В Google Colab установите: !apt-get install -y xvfb")
        print("   3. Убедитесь, что у вас есть графический интерфейс")
        
        # Попытка альтернативного запуска
        if 'google.colab' in sys.modules:
            print("\n🔄 Попытка альтернативного запуска для Google Colab...")
            try:
                # Устанавливаем переменную DISPLAY и пробуем снова
                os.environ['DISPLAY'] = ':99'
                print("🔧 Установлена переменная DISPLAY=:99")
                
                # Небольшая задержка для инициализации
                import time
                time.sleep(2)
                
                root = tk.Tk()
                app = VoiceClonerXTTSApp(root)
                root.mainloop()
                return 0
            except Exception as e2:
                print(f"❌ Альтернативный запуск также не удался: {e2}")
                print("💡 Для работы в Google Colab выполните:")
                print("   !apt-get update && apt-get install -y xvfb")
                print("   !pip install pyvirtualdisplay")
                print("   Затем перезапустите скрипт")
        
        return 1
    return 0

if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        sys.exit(exit_code)
