import streamlit as st
from PIL import Image
import requests

st.set_page_config(page_title="Распознавание руд")

st.title("Распознавание руд")
st.write("Загрузите изображение руды")

classes = [
    "Coal",
    "Diamond",
    "Emerald",
    "Gold",
    "Iron"
]

uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Загруженное изображение", width=250)

    if st.button("Определить"):
        with st.spinner("Модель думает..."):
            try:
                files = {
                    "image": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()

                    pred = result.get("Answer")

                    class_name = classes[int(pred)]

                    st.success(f"Это: {class_name} ")

                else:
                    st.error(f"Ошибка сервера: {response.status_code}")
                    st.write(response.json())

            except Exception as e:
                st.error(f"Ошибка подключения к backend: {e}")