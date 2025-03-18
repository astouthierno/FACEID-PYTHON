import cv2
import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Configuration des chemins relatifs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')

# Chemins des ressources
MODEL_PATH = os.path.join(MODEL_DIR, 'mon_modele.keras')
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
STYLE_IMAGE = os.path.join(BASE_DIR, 'dalein.jpeg')
ABOUT_IMAGE = os.path.join(BASE_DIR, 'astou.jpeg')

# --- Vérification des fichiers critiques ---
missing_files = []
for path in [MODEL_PATH, CASCADE_PATH, STYLE_IMAGE, ABOUT_IMAGE]:
    if not os.path.exists(path):
        missing_files.append(path)

if missing_files:
    st.error(f"Fichiers manquants : {', '.join(missing_files)}")
    st.stop()

# --- Configuration de l'interface ---
st.set_page_config(
    page_title="FaceID System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS intégré
st.markdown(f"""
<style>
    .main {{
        background: url('{STYLE_IMAGE}');
        background-size: cover;
    }}
    .sidebar .sidebar-content {{
        background-color: rgba(255, 255, 255, 0.95) !important;
    }}
    .prediction-box {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# --- Chargement des modèles ---
@st.cache_resource
def load_models():
    return {
        'face_detector': cv2.CascadeClassifier(CASCADE_PATH),
        'ai_model': load_model(MODEL_PATH, compile=False)
    }

models = load_models()

# --- Interface sidebar ---
with st.sidebar:
    st.image(ABOUT_IMAGE, use_container_width=True)
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🎥 Reconnaissance", "📚 Documentation"],
        index=0
    )
    
    st.markdown("---")
    st.header("Paramètres")
    detection_scale = st.slider("Sensibilité de détection", 1.01, 1.3, 1.1)
    min_face_size = st.slider("Taille minimale du visage", 50, 200, 100)

# --- Pages ---
if page == "🏠 Accueil":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(ABOUT_IMAGE, caption="Présentation du système")
    
    with col2:
        st.title("FaceID - Système Intelligent de Reconnaissance Faciale")
        st.markdown("""
        **Fonctionnalités clés :**
        - 🚀 Détection en temps réel
        - 🔍 Reconnaissance précise
        - ⚙️ Paramètres ajustables
        - 📊 Feedback visuel

        **Technologies utilisées :**
        - TensorFlow/Keras
        - OpenCV
        - Streamlit
        """)

elif page == "🎥 Reconnaissance":
    st.title("Mode Reconnaissance")
    st.write("Positionnez-vous face à la caméra pour l'identification")

    # Initialisation webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Contrôles
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Démarrer la reconnaissance")
    with col2:
        stop_btn = st.button("Arrêter")

    # Boucle de traitement
    if start_btn and not stop_btn:
        frame_placeholder = st.empty()
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur d'accès à la webcam")
                break

            # Détection des visages
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = models['face_detector'].detectMultiScale(
                gray,
                scaleFactor=detection_scale,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size)
            )

            # Traitement des visages
            for (x, y, w, h) in faces:
                try:
                    face_img = cv2.resize(frame[y:y+h, x:x+w], (100, 100))
                    face_array = img_to_array(face_img) / 255.0
                    pred = models['ai_model'].predict(np.expand_dims(face_array, axis=0))[0]
                    
                    # Affichage des résultats
                    confidence = np.max(pred) * 100
                    label = "Astou Diallo" if confidence > 75 else "Inconnu"
                    color = (0, 255, 0) if label == "Astou Diallo" else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, 
                               f"{label} ({confidence:.1f}%)",
                               (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, color, 2)
                
                except Exception as e:
                    st.error(f"Erreur de prédiction : {str(e)}")

            # Affichage du flux
            frame_placeholder.image(frame, channels="BGR")

        cap.release()

elif page == "📚 Documentation":
    st.title("Documentation Technique")
    st.markdown("""
    ## Structure du projet
    ```
    FACEID-PYTHON/
    ├── models/
    │   └── mon_modele.keras
    ├── src/
    │   ├── app.py
    │   └── haarcascade_frontalface_default.xml
    └── static/
        ├── style_image.jpg
        └── about_image.jpg
    ```
    
    ## Dépendances
    ```bash
    pip install tensorflow opencv-python-headless streamlit
    ```
    """)

# Gestion de la fermeture
#if stop_btn:
    #st.experimental_rerun()