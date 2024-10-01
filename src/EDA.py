import matplotlib
matplotlib.use('Agg')  # Utilise un backend non interactif

import os
import matplotlib.pyplot as plt

def display_sample_images(data_dir, num_images=20):

    """Affiche un échantillon d'images du jeu de données."""

    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.flatten()

    count = 0
    for label in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, label)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                axes[count].imshow(plt.imread(img_path), cmap='gray')
                axes[count].set_title(label)
                axes[count].axis('off')
                count += 1
                if count >= num_images:
                    break
        if count >= num_images:
            break

    plt.tight_layout()
    plt.savefig('sample_images.png')  # Sauvegarde le graphique
    plt.close(fig)

if __name__ == "__main__":
    display_sample_images("../data/raw")
