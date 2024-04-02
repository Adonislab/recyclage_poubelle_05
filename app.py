# Import des bibliothèques nécessaires
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

# Charger le modèle avec une fonction memoisée
@st.cache(allow_output_mutation=True)
def load_model_from_disk():
    model_path = "Model-RESNET-Classification-dechets.h5"
    model = load_model(model_path, compile=False)
    return model

# Chargement du modèle
model = load_model_from_disk()

# Informations sur le projet
st.sidebar.title("Informations sur le projet")
st.sidebar.write("Nom: Tanguy Adonis NOBIME")
st.sidebar.write("But du projet:")
st.sidebar.write('''Ce projet vise à reconnaître si un objet doit être placé dans une poubelle spécifique. Il utilise l'IA pour classer les objets en carton, masques, métal, papier ou plastique.''')
st.sidebar.write("Nombre de classes: 5")
st.sidebar.write("Classes: Carton, Masques, Metal, Papier, Plastiques")
st.sidebar.write("Contact: +22951518759")
st.sidebar.write("Lien: www.linkedin.com/in/tanguy-adonis-nobime-078166200")

# Fonction de prédiction
def predict_waste_category(image):
    img = Image.open(image)
    img = np.asarray(img)
    img_resized = np.array(Image.fromarray(img).resize((256, 256)))
    img_resized = np.expand_dims(img_resized, axis=0)
    pred = model.predict(img_resized)
    predicted_class_index = np.argmax(pred)
    return predicted_class_index

# Interface utilisateur
st.title("Poubelle Intelligente : Détection d'un objet recyclable")

uploaded_image = st.file_uploader("Téléchargez l'image de l'objet à classer", type=['png', 'jpeg', 'jpg'])

if uploaded_image:
    predicted_class_index = predict_waste_category(uploaded_image)
    
    # Affichage de l'image redimensionnée
    st.image(uploaded_image, caption="Image téléchargée (redimensionnée)", use_column_width=True)
    
    # Affichage de la prédiction
    waste_categories = ['Cartons', 'Masques', 'Métaux', 'Papier', 'Plastiques']
    st.write(f"Je pense que l'objet est à mettre dans la poubelle pour des {waste_categories[predicted_class_index]}")
