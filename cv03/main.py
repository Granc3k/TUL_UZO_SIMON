# made by Martin "Granc3k" Šimon, Jakub "Parrot2" Keršláger

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    radians = math.radians(angle)

    # Matice rotace
    rotation_matrix = np.array(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ]
    )

    # Calc new dims image
    corners = np.array(
        [
            [-width // 2, -height // 2],
            [width // 2, -height // 2],
            [width // 2, height // 2],
            [-width // 2, height // 2],
        ]
    )

    new_corners = np.dot(corners, rotation_matrix.T)
    new_width = int(np.ceil(new_corners[:, 0].max() - new_corners[:, 0].min()))
    new_height = int(np.ceil(new_corners[:, 1].max() - new_corners[:, 1].min()))

    # New střed image
    new_center = np.array([new_width // 2, new_height // 2])
    original_center = np.array([width // 2, height // 2])

    # Create blank image s black pozadím
    rotated_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Inverz. matice rotace pro zpětné mapování souřadnic
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)

    # Iterace přes každý pixel v new obrázku
    for i in range(new_height):
        for j in range(new_width):
            original_coords = np.dot(
                inv_rotation_matrix, np.array([i - new_center[1], j - new_center[0]])
            )
            x, y = original_coords + original_center

            if 0 <= int(y) < width and 0 <= int(x) < height:
                rotated_image[i, j] = image[int(x), int(y)]

    return rotated_image


def main():
    # Load obrázku - matplotlib
    image = mpimg.imread("./data/cv03_robot.bmp")

    # Input úhlu
    angle = float(input("Zadejte úhel otočení (kladný doleva, záporný doprava): "))

    # Rotate obrázku
    rotated = rotate_image(image, angle)

    # Plot výsledku
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Původní obrázek")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(rotated)
    plt.title(f"Otočený o {angle}°")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
