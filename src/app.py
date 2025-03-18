import cv2
import numpy as np
import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Configuration de base
st.set_page_config(
    page_title="FaceID - Reconnaissance Faciale",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chemins des ressources (à adapter) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mon_modele.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
STYLE_IMAGE = os.path.join(BASE_DIR, "static", "style_image.jpg")
ABOUT_IMAGE = os.path.join(BASE_DIR, "static", "about_image.jpg")

# --- Style CSS personnalisé ---
st.markdown("""
<style>
    .main {background: url('https://img.freepik.com/photos-premium/fond-ecran-abstrait-technologie-reconnaissance-faciale_76964-5033.jpg')}
    .sidebar .sidebar-content {background-color: rgba(255, 255, 255, 0.9)!important;}
    h1 {color: #2b5876!important;}
    .stButton>button {background: linear-gradient(45deg, #4e54c8, #8f94fb)!important;}
    .prediction-box {padding: 20px; border-radius: 10px; background: rgba(255,255,255,0.9); margin: 10px;}
</style>
""", unsafe_allow_html=True)

# --- Chargement des ressources ---
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)

@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Modèle introuvable ! Vérifiez le chemin du modèle.")
        st.stop()
    return load_model(MODEL_PATH, compile=False)

# --- Interface sidebar ---
with st.sidebar:
    st.image(STYLE_IMAGE, use_column_width=True)
    st.title("Navigation")
    page = st.radio("", ["🏠 Accueil", "👤 Reconnaissance", "ℹ️ À propos"])

    st.markdown("---")
    st.markdown("""
    **Paramètres avancés :**
    """)
    detection_confidence = st.slider("Seuil de confiance", 1.0, 1.3, 1.1, 0.05)
    show_fps = st.checkbox("Afficher les FPS", True)

# --- Pages ---
if page == "🏠 Accueil":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(ABOUT_IMAGE, use_column_width=True)
    
    with col2:
        st.title("Bienvenue dans FaceID")
        st.markdown("""
        Un système intelligent de reconnaissance faciale développé avec :
        - 🧠 Réseaux de neurones profonds (CNN)
        - 👁 Vision par ordinateur (OpenCV)
        - 🚀 Interface moderne (Streamlit)

        **Fonctionnalités clés :**
        ✔️ Détection en temps réel  
        ✔️ Reconnaissance multi-visages  
        ✔️ Interface utilisateur intuitive  
        ✔️ Paramètres personnalisables
        """)
        st.success("Sélectionnez 'Reconnaissance' dans le menu pour commencer !")

elif page == "👤 Reconnaissance":
    st.title("Reconnaissance Faciale en Temps Réel")
    st.caption("Activez votre webcam et positionnez-vous face à la caméra")

    # Initialisation des composants
    model = load_ai_model()
    face_cascade = load_face_detector()
    FRAME_WINDOW = st.image([])
    stop_button = st.button("Arrêter la reconnaissance")

    # Classes du modèle (à adapter)
    CLASS_NAMES = {
        0: "Utilisateur Inconnu",
        1: "Astou Diallo"
    }

    # Démarrer la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Boucle de traitement
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Problème d'accès à la webcam")
            break

        # Traitement de l'image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=detection_confidence,
            minNeighbors=5,
            minSize=(100, 100)
        )

        # Détection des visages
        for (x, y, w, h) in faces:
            try:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100))
                face_array = img_to_array(face_img) / 255.0
                face_array = np.expand_dims(face_array, axis=0)
                
                pred = model.predict(face_array)
                label = CLASS_NAMES[np.argmax(pred)]
                confidence = np.max(pred) * 100

                # Affichage des résultats
                color = (0, 255, 0) if confidence > 75 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label} ({confidence:.1f}%)", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, color, 2)
                
            except Exception as e:
                st.error(f"Erreur de prédiction : {str(e)}")

        # Affichage du flux vidéo
        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

elif page == "ℹ️ À propos":
    st.title("À propos de FaceID")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Version :** 1.0.0  
        **Développeur :** Astou Diallo  
        **Technologies :**
        - Python 3.10
        - TensorFlow 2.12
        - OpenCV 4.7
        - Streamlit 1.22

        **Fonctionnement :**
        Le système utilise un réseau de neurones convolutifs (CNN)
        pré-entraîné sur un dataset de visages pour réaliser la 
        reconnaissance en temps réel.
        """)
    
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*FZV2m8x2sKkq2O7X8FzeeA.jpeg", 
                caption="Architecture du modèle de reconnaissance")

    st.markdown("---")
    st.success("Projet académique - Master IA 2023")
