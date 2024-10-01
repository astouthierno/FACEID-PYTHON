import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Charger le modèle
model = load_model('/home/santoudllo/Desktop/astou/FACEID-PYTHON/models/mon_modele.keras')

# Définir les classes (selon ton modèle)
class_names = ['Classe1', 'Classe2', 'Classe3', 'Classe4', 'Classe5', 'Classe6', 'Classe7']

def predict_image(image):
    # Prétraitement de l'image
    image = cv2.resize(image, (100, 100))  # Redimensionner l'image
    image = img_to_array(image)  # Convertir en tableau numpy
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour le batch
    image /= 255.0  # Normaliser
    prediction = model.predict(image)  # Faire la prédiction
    return class_names[np.argmax(prediction)]  # Retourner la classe prédite

# Configuration de l'interface
st.title("Reconnaissance Faciale")
st.write("Téléchargez une image pour effectuer la reconnaissance faciale.")

# Uploader une image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire et afficher l'image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Image téléchargée.', use_column_width=True)
    
    # Prédire et afficher le résultat
    prediction = predict_image(image)
    st.write(f"Classe prédite : {prediction}")
