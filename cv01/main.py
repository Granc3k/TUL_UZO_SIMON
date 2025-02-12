# made by Martin "Granc3k" Šimon, Jakub "Parrot2" Keršláger
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_images_from_folder(folder):
    # Load obrázků z folderu
    images = {}
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images[filename] = img
    return images


def compute_histogram(image):
    # Výpočet histogramu z grayscalu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # normalizace histogramu
    cv2.normalize(hist, hist)
    return hist


def compare_histograms(hist1, hist2):
    # porovnání 2 histogramů pomocí bhattacharyya vzdálenosti
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def sort_images(images, reference_image):
    # ref histogram
    reference_hist = compute_histogram(reference_image)
    distances = []
    # porovnání všech obrázků s referenčním
    for name, img in images.items():
        hist = compute_histogram(img)
        distance = compare_histograms(reference_hist, hist)
        distances.append((name, distance, img))
    # sort přes lambdu
    distances.sort(key=lambda x: x[1])
    return distances


def display_sorted_images_grid(images):

    image_names = list(images.keys())
    num_images = len(image_names)
    fig, axes = plt.subplots(num_images, num_images, figsize=(15, 15))

    for i, ref_name in enumerate(image_names):
        reference_image = images[ref_name]
        sorted_images = sort_images(images, reference_image)

        for j, (name, score, img) in enumerate(sorted_images):
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Print score histogramu pro validaci
            ax.set_title(f"Score: {score:.4f}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# Main fce
def main():
    folder_path = "./data/"
    images = load_images_from_folder(folder_path)

    if images:
        display_sorted_images_grid(images)
    else:
        print("Nic to nenalezlo")


if __name__ == "__main__":
    main()
