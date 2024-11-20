import streamlit as st
from fastai.vision.all import *
import plotly.express as px
from PIL import Image
import numpy as np
import pathlib
#uzgarish
# Streamlit sarlavhasi
st.title('Rasmlarni klassifikatsiya qiluvchi model')
st.write("Klassar : Car Airplane Boat Bird Flower Tree Vegetable Fruit Person Cosmetics")
# Fayl yo‘lini aniqlash
model_path = pathlib.Path('transport_model.pkl')

# Rasm yuklash
files = st.file_uploader("Rasm yuklash", type=["avif", "png", "jpeg", "gif", "svg", "jfif"])
if files:
    # Rasmni yuklash va ko‘rsatish
    st.image(files)

    # Tasvirni RGB formatga o‘tkazish
    try:
        image = Image.open(files).convert("RGB")  # Tasvirni ochish va RGB formatga aylantirish
        img = PILImage.create(np.array(image))  # Tasvirni FastAI formatiga aylantirish
    except Exception as e:
        st.error(f"Tasvirni ishlov berishda xatolik: {e}")
        img = None

    if img is not None:
        try:
            # Modelni yuklash
            model = load_learner(model_path)
        except Exception as e:
            st.error(f"Modelni yuklashda xatolik: {e}")
            model = None

        if model is not None:
            try:
                # Bashoratni topish
                pred, pred_id, probs = model.predict(img)
                st.success(f"Bashorat: {pred}")
                st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")

                # Diagrammani chizish
                fig = px.bar(
                    x=probs * 100, 
                    y=model.dls.vocab, 
                    labels={'x': "Ehtimollik (%)", 'y': "Klasslar"}, 
                    orientation='h'
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Bashoratda xatolik: {e}")


# import streamlit as st
# from fastai.vision.all import *
# import plotly.express as px
# import pathlib
# pathlib.PosixPath = pathlib.Path

# # title
# st.title('Transportni klassifikatsiya qiluvchi model')

# # Rasmni joylash
# files = st.file_uploader("Rasm yuklash", type=["avif", "png", "jpeg", "gif", "svg"])
# if files:
#     st.image(files)  # rasmni chiqarish
#     # PIL convert
#     img = PILImage.create(files)
    
#     # Modelni yuklash
#     model = load_learner('transport_model.pkl')

#     # Bashorat qiymatni topamiz
#     pred, pred_id, probs = model.predict(img)
#     st.success(f"Bashorat: {pred}")
#     st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")

#     # Plotting
#     fig = px.bar(x=probs * 100, y=model.dls.vocab)
#     st.plotly_chart(fig)