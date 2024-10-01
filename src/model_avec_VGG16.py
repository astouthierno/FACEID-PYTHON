import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_model(num_classes):
    """Crée le modèle VGG16 avec des couches supplémentaires pour la classification."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # Geler les couches de base
    for layer in base_model.layers:
        layer.trainable = False

    return model

def plot_predictions(images, predictions, labels, output_dir):
    """Sauvegarde les images avec leurs prédictions et labels réels."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for img, pred, label, ax in zip(images, predictions, labels, axes):
        ax.imshow(img)
        ax.set_title(f'Pred: {pred}\nTrue: {label}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    plt.close(fig)  # Ferme la figure pour éviter d'afficher une figure vide

def train_model(data_dir, output_dir):
    """Entraîne le modèle et l'évalue sur un ensemble de test."""
    # Générateur de données pour l'entraînement
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = create_model(num_classes=len(train_generator.class_indices))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    model.fit(train_generator, validation_data=validation_generator, epochs=10)

    # Sauvegarde du modèle au format natif Keras
    model.save(os.path.join(output_dir, 'mon_modele.keras'))

    # Évaluation sur l'ensemble de test
    test_loss, test_accuracy = model.evaluate(validation_generator)
    print(f'Test accuracy: {test_accuracy}, Test loss: {test_loss}')

    # Réalisation de prédictions sur quelques images du jeu de validation
    validation_images, validation_labels = next(validation_generator)
    predictions = model.predict(validation_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(validation_labels, axis=1)

    # Visualisation des résultats
    plot_predictions(validation_images, predicted_classes, true_classes, output_dir)

if __name__ == "__main__":
    data_directory = "../data/processed"  # Répertoire contenant les images prétraitées
    output_directory = "../models"  # Répertoire où le modèle sera sauvegardé
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_model(data_directory, output_directory)
