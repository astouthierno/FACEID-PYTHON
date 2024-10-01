from sklearn.datasets import fetch_lfw_people
import os
import matplotlib.pyplot as plt

def download_lfw_dataset(data_dir="data"):
    
    """Télécharge le jeu de données LFW et le stocke dans un répertoire local"""
    
    # Téléchargement des images

    print("Téléchargement du jeu de données LFW...")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    images = lfw_people.images
    labels = lfw_people.target
    target_names = lfw_people.target_names

    # Création du répertoire de sauvegarde

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Création du répertoire: {data_dir}")

    # Sauvegarde des images

    print("Sauvegarde des images dans le dossier...")

    for i, img in enumerate(images):
        label = target_names[labels[i]]
        person_dir = os.path.join(data_dir, label)

        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            print(f"Création du répertoire pour {label}")

        img_path = os.path.join(person_dir, f"{i}.jpg")
        plt.imsave(img_path, img, cmap='gray')

        # Confirmation de sauvegarde

        if i % 10 == 0:  
            print(f"Sauvegardé: {img_path}")

    # Confirmation de sauvegarde

    print(f"\nDataset téléchargé et sauvegardé dans {data_dir}")
    print(f"Nombre total d'images: {len(images)}")

if __name__ == "__main__":
    
    download_lfw_dataset("../FACEID-PYTHON/data/raw")
