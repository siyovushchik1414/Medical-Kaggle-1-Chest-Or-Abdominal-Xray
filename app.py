import streamlit as st
from PIL import Image
# import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from pathlib import Path
import GD_download

def load_model_custom():
    import requests

    # Получите ID файла из ссылки
    file_id = "1OxdfQQN3mAIJOpLRF-ZBL5Zk8wJcI42m"

    # Загрузите файл
    response = requests.get(
        f"https://drive.google.com/uc?export=download&id={file_id}"
    )
    open("model.h5", "wb").write(response.content)

    # Загрузите модель в память
    model = load_model("model.h5")
    return model


model = load_model_custom()
img_size = 150
  

# def predict(name):
#     image = st.file_uploader("Загрузите фотографию " + name, type=["png", "jpg", "jpeg"], )
#     if image:
#         st.image(image=image)
#         im = Image.open(image)
#         im.filename = image.name
#         SamplePhoto = np.asarray(im)
#         resized_arr = cv2.resize(SamplePhoto, (img_size, img_size))

#         data = []
#         data.append([resized_arr, 0])
#         data = np.array(data, dtype = object)
#         SamplePhotoXTrain = []
#         SamplePhotoYTrain = []
#         for feature, label in data:
#             SamplePhotoXTrain.append(np.array(feature))
#             SamplePhotoYTrain.append(np.array(label))

#         SamplePhotoXTrain = np.array(SamplePhotoXTrain) / 255
#         SamplePhotoXTrain = SamplePhotoXTrain.reshape(-1, img_size, img_size, 1)
#         Prediction = NeuralNetwork.predict(SamplePhotoXTrain)
#         FloatNumber = (1.0 - Prediction[0][0]) * 100
#         ANS = str("%.2f" % FloatNumber)
#         if FloatNumber > 60:
#           st.markdown("<h4 style='text-align: center; color: white;'>Обнаружены признаки пневмонии.</h4>", unsafe_allow_html=True)
#           st.markdown(f"<h5 style='text-align: center; color: white;'>Вероятность наличия составляет {ANS}%</h5>", unsafe_allow_html=True)
#         else:
#           st.markdown("<h4 style='text-align: center; color: white;'>Признаков заболевания не обнаружено.</h4>", unsafe_allow_html=True)
#           st.markdown(f"<h5 style='text-align: center; color: white;'>Вероятность наличия составляет {ANS}%</h5>", unsafe_allow_html=True)
      
        

def main():
  st.markdown("<h2 style='text-align: center; color: white;'>Модель машинного обучения для классификации изображений рентгена грудной клетки и брюшной полости</h2>", unsafe_allow_html=True)
  st.image('https://th.bing.com/th/id/OIG.a2P2Ck24DaWtlqyUcrCt?pid=ImgGn', caption='“ Высшее благо достигается на основе полного физического и умственного здоровья. ” —Цицерон', use_column_width=True)
  st.markdown("<h6 style='text-align: center; color: white;'>Загрузите фотографию, чтобы передать её модели искусственного интеллекта</h6>", unsafe_allow_html=True)
  # predict('image')  # Передаем имя загруженного файла  

if __name__ == "__main__":
  main()
