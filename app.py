import streamlit as st
from faster_whisper import WhisperModel
import os

# 1. Настройка внешнего вида страницы
st.set_page_config(page_title="Транскриптор Pro", page_icon="🎙️", layout="wide")

st.title("🎙️ Локальна транскрипцiя: UA / RU / EN")
st.markdown("""
Эта программа автоматически определяет язык (украинский, русский или английский) 
и превращает аудио из видео в текст. Всё работает локально на твоём ПК.
""")

# 2. Загрузка модели (кешируем, чтобы не грузить каждый раз)
@st.cache_resource
def load_model():
    # Используем "base" для баланса скорости и качества. 
    # Если будет медленно, можно заменить на "tiny".
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# 3. Боковая панель с информацией
with st.sidebar:
    st.header("Налаштування та iнфо")
    st.info("Модель: Whisper Base")
    st.write("Пiдтримка автовиявлення 100+ мов, включаючи основi.")

# 4. Основной функционал загрузки
uploaded_file = st.file_uploader("Обери вiдеофайл", type=["mp4", "mkv", "avi", "mov"])

if uploaded_file is not None:
    # Сохраняем видео во временный файл
    temp_filename = "temp_process_video.mp4"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.video(uploaded_file)

    with col2:
        if st.button("🚀 Почати розшифровку", use_container_width=True):
            try:
                with st.spinner("Аналiзую аудiо..."):
                    # Транскрипция с автоопределением языка
                    segments, info = model.transcribe(temp_filename, beam_size=5)
                    
                    st.success(f"Мову виявлено: {info.language.upper()} (уверенность: {info.language_probability:.2%})")
                    
                    # Собираем текст и выводим его в реальном времени
                    full_text = ""
                    text_area = st.empty() # Место для динамического обновления текста
                    
                    result_content = ""
                    for segment in segments:
                        timestamp = f"[{int(segment.start // 60):02d}:{int(segment.start % 60):02d}]"
                        line = f"{timestamp} {segment.text}\n"
                        result_content += line
                        # Обновляем блок текста на лету
                        text_area.text_area("Розпiзнае текст:", result_content, height=400)
                    
                    # Кнопка скачивания появится после завершения
                    st.download_button(
                        label="💾 Скачать результат (.txt)",
                        data=result_content,
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Произошла ошибка: {e}")
            finally:
                # Всегда удаляем временный файл после обработки
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

# Инструкция внизу, если файл не выбран
else:
    st.info("Загрузи видео, чтобы начать работу.")