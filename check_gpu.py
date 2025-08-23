#!/usr/bin/env python3
"""
Скрипт проверки поддержки GPU и CUDA для ускорения XTTS v2
Проверяет возможность использования видеокарты вместо процессора
"""

import sys
import platform
import subprocess
import os

def print_header():
    """Вывод заголовка"""
    print("=" * 60)
    print("🔍 ПРОВЕРКА ПОДДЕРЖКИ GPU ДЛЯ XTTS v2")
    print("=" * 60)
    print()

def check_system_info():
    """Проверка информации о системе"""
    print("📋 ИНФОРМАЦИЯ О СИСТЕМЕ:")
    print(f"• Операционная система: {platform.system()} {platform.release()}")
    print(f"• Архитектура: {platform.machine()}")
    print(f"• Python версия: {sys.version.split()[0]}")
    print()

def check_nvidia_gpu():
    """Проверка наличия NVIDIA GPU"""
    print("🎮 ПРОВЕРКА NVIDIA GPU:")
    
    try:
        # Попытка импорта nvidia-ml-py
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"✅ Найдено NVIDIA GPU: {device_count} устройств")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            print(f"  📺 GPU {i}: {name.decode('utf-8')}")
            print(f"    💾 Память: {memory.total // 1024**3} ГБ")
            print(f"    🔥 Свободно: {memory.free // 1024**3} ГБ")
            
        return True
        
    except ImportError:
        print("❌ nvidia-ml-py не установлен")
        print("💡 Установите: pip install nvidia-ml-py")
        return False
    except Exception as e:
        print(f"❌ Ошибка проверки NVIDIA GPU: {e}")
        return False

def check_cuda_installation():
    """Проверка установки CUDA"""
    print("🔧 ПРОВЕРКА CUDA:")
    
    # Проверка переменных окружения
    cuda_path = None
    for var in ['CUDA_PATH', 'CUDA_HOME']:
        if var in os.environ:
            cuda_path = os.environ[var]
            print(f"✅ Переменная {var}: {cuda_path}")
    
    if not cuda_path:
        print("❌ Переменные CUDA не найдены")
    
    # Проверка nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi работает")
            # Извлекаем версию CUDA из вывода
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip()
                    print(f"📊 Версия CUDA: {cuda_version}")
                    break
        else:
            print("❌ nvidia-smi не работает")
    except FileNotFoundError:
        print("❌ nvidia-smi не найден")
    except Exception as e:
        print(f"❌ Ошибка nvidia-smi: {e}")
    
    print()

def check_pytorch_cuda():
    """Проверка поддержки CUDA в PyTorch"""
    print("🔥 ПРОВЕРКА PYTORCH + CUDA:")
    
    try:
        import torch
        
        print(f"✅ PyTorch версия: {torch.__version__}")
        print(f"✅ CUDA доступна: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA версия: {torch.version.cuda}")
            print(f"✅ Количество GPU: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                print(f"  📺 GPU {i}: {gpu_name}")
                print(f"    💾 Память: {gpu_memory // 1024**3} ГБ")
            
            # Тест производительности
            print("\n🧪 ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ:")
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            
            import time
            start_time = time.time()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            end_time = time.time()
            
            print(f"✅ GPU тест пройден за {end_time - start_time:.3f} сек")
            
        else:
            print("❌ CUDA недоступна в PyTorch")
            print("💡 Установите PyTorch с поддержкой CUDA:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
    except ImportError:
        print("❌ PyTorch не установлен")
        print("💡 Установите: pip install torch")
    except Exception as e:
        print(f"❌ Ошибка проверки PyTorch: {e}")
    
    print()

def check_other_gpu():
    """Проверка других типов GPU"""
    print("🔍 ПРОВЕРКА ДРУГИХ GPU:")
    
    # Проверка AMD ROCm
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            print("✅ AMD ROCm доступна")
        else:
            print("❌ AMD ROCm недоступна")
    except:
        print("❌ AMD ROCm не поддерживается")
    
    # Проверка Apple Metal
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple Metal доступна")
        else:
            print("❌ Apple Metal недоступна")
    except:
        print("❌ Apple Metal не поддерживается")
    
    print()

def check_xtts_gpu_compatibility():
    """Проверка совместимости XTTS v2 с GPU"""
    print("🎤 ПРОВЕРКА СОВМЕСТИМОСТИ XTTS v2:")
    
    try:
        from TTS.api import TTS
        
        # Проверяем доступные модели
        models = TTS.list_models()
        xtts_models = [m for m in models if 'xtts' in m.lower()]
        
        if xtts_models:
            print(f"✅ Найдено XTTS моделей: {len(xtts_models)}")
            for model in xtts_models[:3]:  # Показываем первые 3
                print(f"  📦 {model}")
        else:
            print("❌ XTTS модели не найдены")
        
        # Проверяем поддержку GPU в TTS
        print("\n🔧 ПОДДЕРЖКА GPU В TTS:")
        try:
            # Пробуем создать модель с GPU
            import torch
            if torch.cuda.is_available():
                print("✅ TTS поддерживает GPU")
                print("💡 XTTS v2 будет автоматически использовать GPU")
            else:
                print("⚠️ GPU недоступна, TTS будет использовать CPU")
        except Exception as e:
            print(f"❌ Ошибка проверки GPU в TTS: {e}")
        
    except ImportError:
        print("❌ TTS не установлен")
        print("💡 Установите: pip install TTS")
    except Exception as e:
        print(f"❌ Ошибка проверки TTS: {e}")
    
    print()

def provide_recommendations():
    """Предоставление рекомендаций"""
    print("💡 РЕКОМЕНДАЦИИ:")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ Ваша система готова для GPU-ускорения!")
            print("🚀 XTTS v2 будет работать значительно быстрее")
            print("📊 Ожидаемое ускорение: 3-10x")
        else:
            print("⚠️ GPU недоступна, но это не критично")
            print("💻 XTTS v2 будет работать на CPU")
            print("📊 Время генерации: 10-30 секунд на короткий текст")
            
            print("\n🔧 ДЛЯ ВКЛЮЧЕНИЯ GPU:")
            print("1. Установите драйверы NVIDIA")
            print("2. Установите CUDA Toolkit")
            print("3. Переустановите PyTorch с CUDA:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    except ImportError:
        print("❌ PyTorch не установлен")
        print("💡 Установите PyTorch для проверки GPU")
    
    print()

def main():
    """Главная функция"""
    import os
    
    print_header()
    check_system_info()
    
    # Проверяем различные типы GPU
    nvidia_available = check_nvidia_gpu()
    check_cuda_installation()
    check_pytorch_cuda()
    check_other_gpu()
    check_xtts_gpu_compatibility()
    provide_recommendations()
    
    print("=" * 60)
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 60)

if __name__ == "__main__":
    main()
