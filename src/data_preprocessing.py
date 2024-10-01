import os
import numpy as np
from skimage import io
from skimage.transform import resize

def preprocess_images(data_dir, output_dir, target_size=(100, 100)):

    """Prétraitement des images : redimensionnement et normalisation."""

    X = []
    y = []
    target_names = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, label)
        if os.path.isdir(person_dir):
            target_names.append(label)

            # Création d'un sous-répertoire pour les images prétraitées

            processed_dir = os.path.join(output_dir, label)
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)

            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = io.imread(img_path) 
                img_resized = resize(img, target_size)
                
                # Normalisation et conversion en uint8

                img_resized = (img_resized * 255).astype(np.uint8)

                X.append(img_resized)
                y.append(label)

                # Sauvegarde l'image prétraitée

                output_path = os.path.join(processed_dir, img_file)
                io.imsave(output_path, img_resized)

    X = np.array(X)
    y = np.array(y)
    
    print(f"Images prétraitées. Nombre total d'images : {len(X)}")
    return X, y, target_names

if __name__ == "__main__":
    data_directory = "../data/raw"  # Répertoire de données d'origine
    output_directory = "../data/processed"  # Répertoire de données pre
    X, y, target_names = preprocess_images(data_directory, output_directory)
