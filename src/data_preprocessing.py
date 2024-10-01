import os
import numpy as np
from skimage import io
from skimage.transform import resize

def preprocess_images(data_dir, target_size=(100, 100)):
    """Prétraitement des images : redimensionnement et normalisation."""
    X = []
    y = []
    target_names = []

    for label in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, label)
        if os.path.isdir(person_dir):
            target_names.append(label)
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = io.imread(img_path)  # Utilise skimage.io pour lire les images
                img_resized = resize(img, target_size)
                X.append(img_resized)
                y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    print(f"Images prétraitées. Nombre total d'images : {len(X)}")
    return X, y, target_names

if __name__ == "__main__":
    data_directory = "../data/raw"  # Répertoire de données d'origine
    X, y, target_names = preprocess_images(data_directory)
