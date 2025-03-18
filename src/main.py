
import nltk
import streamlit as st
import speech_recognition as sr
from nltk.chat.util import Chat, reflections

st.title("Hello World")
import streamlit as st

# Saisie texte
#user_input = st.text_input("Entrez votre texte")
#st.write(f"Vous avez saisi : {user_input}")

#st.markdown("##  l’apprentissage profond et des réseaux neuronaux")
#st.write("Voici une description de votre application.")

#if st.button('Cliquez ici'):
    #st.write("Vous avez cliqué sur le bouton !")
import streamlit as st
import nltk
from nltk.chat.util import Chat, reflections
import speech_recognition as sr

# Fonction pour charger les paires de questions/réponses
def load_pairs_from_file(filename):
    pairs = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):  # Lire les questions/réponses par paires
                question = lines[i].strip()
                answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                pairs.append([question, [answer]])
        return pairs
    except FileNotFoundError:
        st.error("Fichier introuvable. Veuillez vérifier le chemin.")
        return []

# Interface utilisateur avec Streamlit
st.title("Hello World")

# Saisie texte de l'utilisateur
user_input = st.text_input("Entrez votre texte")
st.write(f"Vous avez saisi : {user_input}")

# Bouton d'interaction
if st.button('Cliquez ici'):
    st.write("Vous avez cliqué sur le bouton !")

# Téléchargement de fichier
uploaded_file = st.file_uploader("Choisissez un fichier", type=["csv", "txt"])
if uploaded_file is not None:
    st.write("Fichier téléchargé :", uploaded_file.name)
    # Charge les paires de questions/réponses depuis le fichier téléchargé
    pairs = load_pairs_from_file(uploaded_file)
    if pairs:
        st.write(f"Nombre de paires chargées : {len(pairs)}")

# Exemple de chat avec NLTK
st.markdown("## L’apprentissage profond et des réseaux neuronaux")
st.write("Voici une description de votre application.")



def load_pairs_from_file(filename):
    pairs = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):  # Lire les questions/réponses par paires
                question = lines[i].strip()
                answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                pairs.append([question, [answer]])
        return pairs
    except FileNotFoundError:
        st.error("Fichier introuvable. Veuillez vérifier le chemin.")
        return []

def load_pairs_from_file(filename):
    pairs = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):  # Lire les questions/réponses par paires
                question = lines[i].strip()
                answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                pairs.append([question, [answer]])
        return pairs
    except FileNotFoundError:
        st.error("Fichier introuvable. Veuillez vérifier le chemin.")
        return []


uploaded_file = st.file_uploader("Choisissez un fichier", type=["csv", "txt"])
if uploaded_file is not None:
    st.write("Fichier téléchargé :", uploaded_file.name)



def load_pairs_from_file(filename):
    pairs = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):  # Lire les questions/réponses par paires
                question = lines[i].strip()
                answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                pairs.append([question, [answer]])
        return pairs
    except FileNotFoundError:
        st.error("Fichier introuvable. Veuillez vérifier le chemin.")
        return []


def load_pairs_from_file(filename):
    pairs = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):  # Lire les questions/réponses par paires
                question = lines[i].strip()
                answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                pairs.append([question, [answer]])
        return pairs
    except FileNotFoundError:
        st.error("Fichier introuvable. Veuillez vérifier le chemin.")
        return []


def load_pairs_from_file(filename):
    pairs = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):  # Lire les questions/réponses par paires
                question = lines[i].strip()
                answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                pairs.append([question, [answer]])
        return pairs
    except FileNotFoundError:
        st.error("Fichier introuvable. Veuillez vérifier le chemin.")
        return []

def load_pairs_from_file(filename):
    pairs = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):  # Lire les questions/réponses par paires
                question = lines[i].strip()
                answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                pairs.append([question, [answer]])
        return pairs
    except FileNotFoundError:
        st.error("Fichier introuvable. Veuillez vérifier le chemin.")
        return []


def load_pairs_from_file(filename):
    pairs = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):  # Lire les questions/réponses par paires
                question = lines[i].strip()
                answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                pairs.append([question, [answer]])
        return pairs
    except FileNotFoundError:
        st.error("Fichier introuvable. Veuillez vérifier le chemin.")
        return []


