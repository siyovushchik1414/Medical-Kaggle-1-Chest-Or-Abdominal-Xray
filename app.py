import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# def load_model():

#     save_dest = Path('model')
#     save_dest.mkdir(exist_ok=True)
    
#     f_checkpoint = Path("model/model.h5")

#     if not f_checkpoint.exists():
#         with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
#             from GD_download import download_file_from_google_drive
#             download_file_from_google_drive("1OxdfQQN3mAIJOpLRF-ZBL5Zk8wJcI42m", f_checkpoint)
    
#     model = keras.models.load_model(f_checkpoint, map_location=device)
#     model.eval()
#     return model

# NeuralNetwork = load_model()
# img_size = 150
  

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
