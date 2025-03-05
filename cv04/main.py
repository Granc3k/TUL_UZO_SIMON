# made by Martin "Granc3k" Šimon, Jakub "Parrot2" Keršláger
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import dctn


# Fce pro load obrázků
def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images[filename] = img
    return images


# Fce pro Calc DCT vektoru z obrázku
def compute_dct_vector(image, R=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Transfer na stupně šedi
    dct_coeffs = dctn(gray, norm="ortho")  # Calc DCT transformace
    return dct_coeffs[
        :R, :R
    ].flatten()  # Select horních R×R koeficientů a jejich vyrovnání do vektoru


# Fce pro porovnání dvou DCT vektorů pomocí Euklidovské vzdálenosti
def compare_dct_vectors(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


# Fce pro sort obrázků podle podobnosti k referenčnímu obrázku
def sort_images(images, reference_image):
    reference_dct = compute_dct_vector(
        reference_image
    )  # DCT vektor referenčního obrázku
    distances = []

    for name, img in images.items():
        dct_vector = compute_dct_vector(img)  # Calc DCT vektoru pro obrázek
        distance = compare_dct_vectors(
            reference_dct, dct_vector
        )  # Calc vzdálenosti k referenčnímu
        distances.append((name, distance, img))

    distances.sort(key=lambda x: x[1])  # Sort podle vzdálenosti
    return distances


# Fce pro print mřížky se seřazenými obrázky
def display_sorted_images_grid(images):
    image_names = list(images.keys())
    num_images = len(image_names)
    fig, axes = plt.subplots(num_images, num_images, figsize=(15, 15))

    for i, ref_name in enumerate(image_names):
        reference_image = images[ref_name]  # Vybrání referenčního obrázku
        sorted_images = sort_images(
            images, reference_image
        )  # Sort obrázků podle referenčního

        for j, (name, score, img) in enumerate(sorted_images):
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Print obrázku
            ax.set_title(f"Score: {score:.4f}")  # Print podobnostního skóre
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# Main fce programu
def main():
    folder_path = "./cv04/data/task_5/"
    images = load_images_from_folder(folder_path)  # Load obrázků

    if images:
        display_sorted_images_grid(images)  # Print seřazených obrázků
    else:
        print("Nic to nenalezlo")


if __name__ == "__main__":
    main()
