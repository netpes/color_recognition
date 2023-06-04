import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

def rgb_to_color_name(rgb):
    try:
        return webcolors.rgb_to_name(rgb)
    except ValueError:
        # Fallback to color distance-based lookup
        colors = webcolors.HTML4_NAMES_TO_HEX
        color_names = list(colors.keys())
        color_values = np.array([webcolors.hex_to_rgb(color) for color in colors.values()])

        # Convert the RGB value to a NumPy array
        rgb_array = np.array(rgb)

        # Calculate color distances

        distances = np.sqrt(np.sum((color_values - rgb_array)**2, axis=1))

        # Find the closest color
        closest_index = np.argmin(distances)
        closest_color_name = color_names[closest_index]

        # Set a threshold for similarity
        threshold = 100  # Adjust this value based on your preference

        if distances[closest_index] <= threshold:
            return closest_color_name
        else:
            return 'Unknown'

def get_main_object_color(image_path, num_colors=3):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image
    pixels = image_rgb.reshape(-1, 3)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the labels and cluster centers
    labels = kmeans.labels_
    colors = kmeans.cluster_centers_

    # Find the most dominant label
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    main_label = unique_labels[np.argmax(label_counts)]

    # Find the most dominant color associated with the main label
    main_color = colors[main_label]

    # Convert the color values to integers
    main_color_int = tuple(main_color.astype(int))

    return main_color_int

# Example usage
print('Insert Image Path: ')
image_path = input()
main_object_color = get_main_object_color(image_path)
color_name = rgb_to_color_name(main_object_color)
print('Main object color: RGB', str(main_object_color))
print('Color Name:', color_name)
