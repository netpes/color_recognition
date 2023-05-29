import cv2
import numpy as np
from sklearn.cluster import KMeans


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

    return main_color.astype(int)


# Example usage


print('Insert Image Path: ')
path = input()
image_path = 'a2.jpg'
main_object_color = get_main_object_color(path)
print('Main object color: RGB', main_object_color)