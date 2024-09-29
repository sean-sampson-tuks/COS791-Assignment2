import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load all images from a folder
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
#         if img is not None:
#             images.append((filename, img))
#     return images

# Function to load all images from the current folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):  # Only process .jpg files
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((filename, img))
    return images

# Otsu’s Method Multilevel Thresholding
def otsu_multilevel_thresholding(image, levels):
    thresholds = []
    current_image = image.copy()

    for _ in range(levels - 1):
        _, thresh = cv2.threshold(current_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholds.append(thresh)
        current_image = np.where(current_image >= thresh, 0, current_image)

    segmented_image = np.zeros_like(image)
    for i, thresh in enumerate(sorted(thresholds)):
        segmented_image[np.where(image >= thresh)] = (i + 1) * (255 // levels)

    return segmented_image, thresholds

# Kapur’s Method Thresholding (maximizes entropy)
def kapur_threshold(image, levels):
    hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])
    hist = hist.astype(np.float32) / hist.sum()

    def entropy(thresh):
        prob1 = hist[:thresh].sum()
        prob2 = hist[thresh:].sum()
        prob1 = prob1 if prob1 > 0 else 1
        prob2 = prob2 if prob2 > 0 else 1
        return -(np.log(prob1) * prob1 + np.log(prob2) * prob2)

    thresholds = []
    for _ in range(levels - 1):
        entropies = [entropy(thresh) for thresh in range(1, 255)]
        best_thresh = np.argmax(entropies)
        thresholds.append(best_thresh)
        image = np.where(image >= best_thresh, 0, image)

    segmented_image = np.zeros_like(image)
    for i, thresh in enumerate(sorted(thresholds)):
        segmented_image[np.where(image >= thresh)] = (i + 1) * (255 // levels)

    return segmented_image, thresholds

# Simulated Annealing for Thresholding Optimization
def simulated_annealing(image, objective_function, levels, initial_temperature=1000, cooling_rate=0.95):
    def perturb(thresholds):
        idx = np.random.randint(0, len(thresholds))
        thresholds[idx] += np.random.randint(-10, 10)
        thresholds = np.clip(thresholds, 0, 255)
        return thresholds

    current_thresholds = np.sort(np.random.randint(0, 255, size=(levels - 1)))
    current_cost = objective_function(image, current_thresholds)
    temperature = initial_temperature

    while temperature > 1:
        new_thresholds = perturb(current_thresholds.copy())
        new_cost = objective_function(image, new_thresholds)

        if new_cost < current_cost or np.random.random() < np.exp((current_cost - new_cost) / temperature):
            current_thresholds = new_thresholds
            current_cost = new_cost

        temperature *= cooling_rate

    return current_thresholds

# Variable Neighbourhood Search for Thresholding Optimization
def vns(image, objective_function, levels):
    current_thresholds = np.sort(np.random.randint(0, 255, size=(levels - 1)))
    best_cost = objective_function(image, current_thresholds)

    def local_search(thresholds):
        for i in range(len(thresholds)):
            for change in [-5, 5]:
                new_thresholds = thresholds.copy()
                new_thresholds[i] += change
                new_thresholds = np.clip(new_thresholds, 0, 255)
                new_cost = objective_function(image, new_thresholds)
                if new_cost < best_cost:
                    return new_thresholds, new_cost
        return thresholds, best_cost

    while True:
        new_thresholds, new_cost = local_search(current_thresholds)
        if new_cost < best_cost:
            current_thresholds = new_thresholds
            best_cost = new_cost
        else:
            break

    return current_thresholds

# Objective function (Otsu)
def otsu_objective(image, thresholds):
    thresholds = [0] + sorted(thresholds) + [255]
    total_var = 0
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i+1])
        region = image[mask]
        total_var += np.var(region) * len(region)
    return total_var

# Objective function (Kapur)
def kapur_objective(image, thresholds):
    thresholds = [0] + sorted(thresholds) + [255]
    total_entropy = 0
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i+1])
        hist, _ = np.histogram(image[mask], bins=256, range=[0, 256])
        hist = hist.astype(np.float32) / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        total_entropy += entropy
    return -total_entropy

# Save segmented image
def save_image(filename, image):
    cv2.imwrite(filename, image)

# Main code to load images and apply methods
def main():
    folder = '.'  # This points to the current folder where your images are located
    images = load_images_from_folder(folder)

    for filename, image in images:
        for k in [2, 3, 4, 5]:
            # Otsu's Method
            segmented_otsu, _ = otsu_multilevel_thresholding(image, k)
            save_image(f'output/{filename}_otsu_k{k}.png', segmented_otsu)

            # Kapur's Method
            segmented_kapur, _ = kapur_threshold(image, k)
            save_image(f'output/{filename}_kapur_k{k}.png', segmented_kapur)

            # Simulated Annealing + Otsu
            sa_thresholds = simulated_annealing(image, otsu_objective, k)
            segmented_sa_otsu = otsu_multilevel_thresholding(image, len(sa_thresholds) + 1)[0]
            save_image(f'output/{filename}_sa_otsu_k{k}.png', segmented_sa_otsu)

            # Variable Neighbourhood Search + Otsu
            vns_thresholds = vns(image, otsu_objective, k)
            segmented_vns_otsu = otsu_multilevel_thresholding(image, len(vns_thresholds) + 1)[0]
            save_image(f'output/{filename}_vns_otsu_k{k}.png', segmented_vns_otsu)

if __name__ == "__main__":
    main()



# Main code to load images and apply methods
# def main():
#     folder = 'images_folder'  # Specify your folder containing the images
#     images = load_images_from_folder(folder)

#     for filename, image in images:
#         for k in [2, 3, 4, 5]:
#             # Otsu's Method
#             segmented_otsu, _ = otsu_multilevel_thresholding(image, k)
#             save_image(f'output/{filename}_otsu_k{k}.png', segmented_otsu)

#             # Kapur's Method
#             segmented_kapur, _ = kapur_threshold(image, k)
#             save_image(f'output/{filename}_kapur_k{k}.png', segmented_kapur)

#             # Simulated Annealing + Otsu
#             sa_thresholds = simulated_annealing(image, otsu_objective, k)
#             segmented_sa_otsu = otsu_multilevel_thresholding(image, len(sa_thresholds) + 1)[0]
#             save_image(f'output/{filename}_sa_otsu_k{k}.png', segmented_sa_otsu)

#             # Variable Neighbourhood Search + Otsu
#             vns_thresholds = vns(image, otsu_objective, k)
#             segmented_vns_otsu = otsu_multilevel_thresholding(image, len(vns_thresholds) + 1)[0]
#             save_image(f'output/{filename}_vns_otsu_k{k}.png', segmented_vns_otsu)

# if __name__ == "__main__":
#     main()
