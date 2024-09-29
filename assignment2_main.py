# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # Function to load all images from a folder
# # def load_images_from_folder(folder):
# #     images = []
# #     for filename in os.listdir(folder):
# #         img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
# #         if img is not None:
# #             images.append((filename, img))
# #     return images

# # # Function to load all images from the current folder
# # def load_images_from_folder(folder):
# #     images = []
# #     for filename in os.listdir(folder):
# #         if filename.endswith(".jpg"):  # Only process .jpg files
# #             img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
# #             if img is not None:
# #                 images.append((filename, img))
# #     return images

# # Function to load all images from the current folder
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         if filename.endswith(".jpg"):  # Only process .jpg files
#             img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
#             if img is not None:
#                 images.append((filename, img))
#                 print(f"Loaded image: {filename}")  # Debug: Print the image loaded
#             else:
#                 print(f"Failed to load image: {filename}")  # Debug: Failed loading
#     return images

# # # Function to create necessary output folders
# # def create_output_folders():
# #     base_folder = 'output'
# #     operations = ['otsu', 'kapur', 'sa_otsu', 'vns_otsu']
# #     for operation in operations:
# #         folder_path = os.path.join(base_folder, operation)
# #         if not os.path.exists(folder_path):
# #             os.makedirs(folder_path)

# # Function to create necessary output folders
# def create_output_folders():
#     base_folder = 'output'
#     operations = ['otsu', 'kapur', 'sa_otsu', 'vns_otsu']
#     for operation in operations:
#         folder_path = os.path.join(base_folder, operation)
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#             print(f"Created folder: {folder_path}")  # Debug: Print folder creation

# # # Otsu’s Method Multilevel Thresholding
# # def otsu_multilevel_thresholding(image, levels):
# #     thresholds = []
# #     current_image = image.copy()

# #     for _ in range(levels - 1):
# #         _, thresh = cv2.threshold(current_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #         thresholds.append(thresh)
# #         current_image = np.where(current_image >= thresh, 0, current_image)

# #     segmented_image = np.zeros_like(image)
# #     for i, thresh in enumerate(sorted(thresholds)):
# #         segmented_image[np.where(image >= thresh)] = (i + 1) * (255 // levels)

# #     return segmented_image, thresholds

# # Otsu’s Method Multilevel Thresholding
# def otsu_multilevel_thresholding(image, levels):
#     thresholds = []
#     current_image = image.copy()

#     for _ in range(levels - 1):
#         thresh_value, _ = cv2.threshold(current_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         thresholds.append(thresh_value)  # Append only the threshold value
#         current_image = np.where(current_image >= thresh_value, 0, current_image)

#     segmented_image = np.zeros_like(image)
#     for i, thresh in enumerate(sorted(thresholds)):
#         segmented_image[np.where(image >= thresh)] = (i + 1) * (255 // levels)

#     return segmented_image, thresholds


# # Kapur’s Method Thresholding (maximizes entropy)
# def kapur_threshold(image, levels):
#     hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])
#     hist = hist.astype(np.float32) / hist.sum()

#     def entropy(thresh):
#         prob1 = hist[:thresh].sum()
#         prob2 = hist[thresh:].sum()
#         prob1 = prob1 if prob1 > 0 else 1
#         prob2 = prob2 if prob2 > 0 else 1
#         return -(np.log(prob1) * prob1 + np.log(prob2) * prob2)

#     thresholds = []
#     for _ in range(levels - 1):
#         entropies = [entropy(thresh) for thresh in range(1, 255)]
#         best_thresh = np.argmax(entropies)
#         thresholds.append(best_thresh)
#         image = np.where(image >= best_thresh, 0, image)

#     segmented_image = np.zeros_like(image)
#     for i, thresh in enumerate(sorted(thresholds)):
#         segmented_image[np.where(image >= thresh)] = (i + 1) * (255 // levels)

#     return segmented_image, thresholds

# # Simulated Annealing for Thresholding Optimization
# def simulated_annealing(image, objective_function, levels, initial_temperature=1000, cooling_rate=0.95):
#     def perturb(thresholds):
#         idx = np.random.randint(0, len(thresholds))
#         thresholds[idx] += np.random.randint(-10, 10)
#         thresholds = np.clip(thresholds, 0, 255)
#         return thresholds

#     current_thresholds = np.sort(np.random.randint(0, 255, size=(levels - 1)))
#     current_cost = objective_function(image, current_thresholds)
#     temperature = initial_temperature

#     while temperature > 1:
#         new_thresholds = perturb(current_thresholds.copy())
#         new_cost = objective_function(image, new_thresholds)

#         if new_cost < current_cost or np.random.random() < np.exp((current_cost - new_cost) / temperature):
#             current_thresholds = new_thresholds
#             current_cost = new_cost

#         temperature *= cooling_rate

#     return current_thresholds

# # Variable Neighbourhood Search for Thresholding Optimization
# def vns(image, objective_function, levels):
#     current_thresholds = np.sort(np.random.randint(0, 255, size=(levels - 1)))
#     best_cost = objective_function(image, current_thresholds)

#     def local_search(thresholds):
#         for i in range(len(thresholds)):
#             for change in [-5, 5]:
#                 new_thresholds = thresholds.copy()
#                 new_thresholds[i] += change
#                 new_thresholds = np.clip(new_thresholds, 0, 255)
#                 new_cost = objective_function(image, new_thresholds)
#                 if new_cost < best_cost:
#                     return new_thresholds, new_cost
#         return thresholds, best_cost

#     while True:
#         new_thresholds, new_cost = local_search(current_thresholds)
#         if new_cost < best_cost:
#             current_thresholds = new_thresholds
#             best_cost = new_cost
#         else:
#             break

#     return current_thresholds

# # Objective function (Otsu)
# def otsu_objective(image, thresholds):
#     thresholds = [0] + sorted(thresholds) + [255]
#     total_var = 0
#     for i in range(len(thresholds) - 1):
#         mask = (image >= thresholds[i]) & (image < thresholds[i+1])
#         region = image[mask]
#         total_var += np.var(region) * len(region)
#     return total_var

# # Objective function (Kapur)
# def kapur_objective(image, thresholds):
#     thresholds = [0] + sorted(thresholds) + [255]
#     total_entropy = 0
#     for i in range(len(thresholds) - 1):
#         mask = (image >= thresholds[i]) & (image < thresholds[i+1])
#         hist, _ = np.histogram(image[mask], bins=256, range=[0, 256])
#         hist = hist.astype(np.float32) / hist.sum()
#         entropy = -np.sum(hist * np.log(hist + 1e-10))
#         total_entropy += entropy
#     return -total_entropy


# # Save segmented image
# def save_image(filename, image):
#     print(f"Saving image to: {filename}")  # Debug: Print saving path
#     cv2.imwrite(filename, image)

# # Main code to load images and apply methods
# def main():
#     folder = './Ass2'  # This points to the current folder where your images are located
#     images = load_images_from_folder(folder)

#     # Check if images are loaded
#     if not images:
#         print("No images found in the folder.")  # Debug: No images case
#         return

#     # Create output folders
#     create_output_folders()

#     for filename, image in images:
#         # Get the filename without the extension
#         name_without_extension = os.path.splitext(filename)[0]
#         print(f"Processing image: {filename}")  # Debug: Print processing image

#         for k in [2, 3, 4, 5]:
#             print(f"Processing k={k}")  # Debug: Print current threshold level

#             # Define the output folder paths for each operation
#             output_folders = {
#                 'otsu': f'output/otsu/{name_without_extension}_otsu_k{k}.png',
#                 'kapur': f'output/kapur/{name_without_extension}_kapur_k{k}.png',
#                 'sa_otsu': f'output/sa_otsu/{name_without_extension}_sa_otsu_k{k}.png',
#                 'vns_otsu': f'output/vns_otsu/{name_without_extension}_vns_otsu_k{k}.png'
#             }

#             # Otsu's Method
#             segmented_otsu, _ = otsu_multilevel_thresholding(image, k)
#             save_image(output_folders['otsu'], segmented_otsu)

#             # Kapur's Method
#             segmented_kapur, _ = kapur_threshold(image, k)
#             save_image(output_folders['kapur'], segmented_kapur)

#             # Simulated Annealing + Otsu
#             sa_thresholds = simulated_annealing(image, otsu_objective, k)
#             segmented_sa_otsu = otsu_multilevel_thresholding(image, len(sa_thresholds) + 1)[0]
#             save_image(output_folders['sa_otsu'], segmented_sa_otsu)

#             # Variable Neighbourhood Search + Otsu
#             vns_thresholds = vns(image, otsu_objective, k)
#             segmented_vns_otsu = otsu_multilevel_thresholding(image, len(vns_thresholds) + 1)[0]
#             save_image(output_folders['vns_otsu'], segmented_vns_otsu)

# if __name__ == "__main__":
#     main()

# # # Save segmented image
# # def save_image(filename, image):
# #     print(f"Saving image to: {filename}")  # 
# #     cv2.imwrite(filename, image)


# # def main():
# #     folder = '.'  # This points to the current folder where your images are located
# #     images = load_images_from_folder(folder)

# #     # Create output folders
# #     create_output_folders()

# #     for filename, image in images:
# #         # Get the filename without the extension
# #         name_without_extension = os.path.splitext(filename)[0]

# #         for k in [2, 3, 4, 5]:
# #             # Define the output folder paths for each operation
# #             output_folders = {
# #                 'otsu': f'output/otsu/{name_without_extension}_otsu_k{k}.png',
# #                 'kapur': f'output/kapur/{name_without_extension}_kapur_k{k}.png',
# #                 'sa_otsu': f'output/sa_otsu/{name_without_extension}_sa_otsu_k{k}.png',
# #                 'vns_otsu': f'output/vns_otsu/{name_without_extension}_vns_otsu_k{k}.png'
# #             }

# #             # Otsu's Method
# #             segmented_otsu, _ = otsu_multilevel_thresholding(image, k)
# #             save_image(output_folders['otsu'], segmented_otsu)

# #             # Kapur's Method
# #             segmented_kapur, _ = kapur_threshold(image, k)
# #             save_image(output_folders['kapur'], segmented_kapur)

# #             # Simulated Annealing + Otsu
# #             sa_thresholds = simulated_annealing(image, otsu_objective, k)
# #             segmented_sa_otsu = otsu_multilevel_thresholding(image, len(sa_thresholds) + 1)[0]
# #             save_image(output_folders['sa_otsu'], segmented_sa_otsu)

# #             # Variable Neighbourhood Search + Otsu
# #             vns_thresholds = vns(image, otsu_objective, k)
# #             segmented_vns_otsu = otsu_multilevel_thresholding(image, len(vns_thresholds) + 1)[0]
# #             save_image(output_folders['vns_otsu'], segmented_vns_otsu)

# # if __name__ == "__main__":
# #     main()




# # Main code to load images and apply methods
# # def main():
# #     folder = '.'  # This points to the current folder where your images are located
# #     images = load_images_from_folder(folder)

# #     # Create output folders
# #     create_output_folders()

# #     for filename, image in images:
# #         for k in [2, 3, 4, 5]:
# #             # Define the output folder paths for each operation
# #             output_folders = {
# #                 'otsu': f'output/otsu/{filename}_otsu_k{k}.png',
# #                 'kapur': f'output/kapur/{filename}_kapur_k{k}.png',
# #                 'sa_otsu': f'output/sa_otsu/{filename}_sa_otsu_k{k}.png',
# #                 'vns_otsu': f'output/vns_otsu/{filename}_vns_otsu_k{k}.png'
# #             }

# #             # Otsu's Method
# #             segmented_otsu, _ = otsu_multilevel_thresholding(image, k)
# #             save_image(output_folders['otsu'], segmented_otsu)

# #             # Kapur's Method
# #             segmented_kapur, _ = kapur_threshold(image, k)
# #             save_image(output_folders['kapur'], segmented_kapur)

# #             # Simulated Annealing + Otsu
# #             sa_thresholds = simulated_annealing(image, otsu_objective, k)
# #             segmented_sa_otsu = otsu_multilevel_thresholding(image, len(sa_thresholds) + 1)[0]
# #             save_image(output_folders['sa_otsu'], segmented_sa_otsu)

# #             # Variable Neighbourhood Search + Otsu
# #             vns_thresholds = vns(image, otsu_objective, k)
# #             segmented_vns_otsu = otsu_multilevel_thresholding(image, len(vns_thresholds) + 1)[0]
# #             save_image(output_folders['vns_otsu'], segmented_vns_otsu)

# # if __name__ == "__main__":
# #     main()


# # Main code to load images and apply methods
# # def main():
# #     folder = '.'  # This points to the current folder where your images are located
# #     images = load_images_from_folder(folder)

# #     for filename, image in images:
# #         for k in [2, 3, 4, 5]:
# #             # Otsu's Method
# #             segmented_otsu, _ = otsu_multilevel_thresholding(image, k)
# #             save_image(f'output/{filename}_otsu_k{k}.png', segmented_otsu)

# #             # Kapur's Method
# #             segmented_kapur, _ = kapur_threshold(image, k)
# #             save_image(f'output/{filename}_kapur_k{k}.png', segmented_kapur)

# #             # Simulated Annealing + Otsu
# #             sa_thresholds = simulated_annealing(image, otsu_objective, k)
# #             segmented_sa_otsu = otsu_multilevel_thresholding(image, len(sa_thresholds) + 1)[0]
# #             save_image(f'output/{filename}_sa_otsu_k{k}.png', segmented_sa_otsu)

# #             # Variable Neighbourhood Search + Otsu
# #             vns_thresholds = vns(image, otsu_objective, k)
# #             segmented_vns_otsu = otsu_multilevel_thresholding(image, len(vns_thresholds) + 1)[0]
# #             save_image(f'output/{filename}_vns_otsu_k{k}.png', segmented_vns_otsu)

# # if __name__ == "__main__":
# #     main()



# # Main code to load images and apply methods
# # def main():
# #     folder = 'images_folder'  # Specify your folder containing the images
# #     images = load_images_from_folder(folder)

# #     for filename, image in images:
# #         for k in [2, 3, 4, 5]:
# #             # Otsu's Method
# #             segmented_otsu, _ = otsu_multilevel_thresholding(image, k)
# #             save_image(f'output/{filename}_otsu_k{k}.png', segmented_otsu)

# #             # Kapur's Method
# #             segmented_kapur, _ = kapur_threshold(image, k)
# #             save_image(f'output/{filename}_kapur_k{k}.png', segmented_kapur)

# #             # Simulated Annealing + Otsu
# #             sa_thresholds = simulated_annealing(image, otsu_objective, k)
# #             segmented_sa_otsu = otsu_multilevel_thresholding(image, len(sa_thresholds) + 1)[0]
# #             save_image(f'output/{filename}_sa_otsu_k{k}.png', segmented_sa_otsu)

# #             # Variable Neighbourhood Search + Otsu
# #             vns_thresholds = vns(image, otsu_objective, k)
# #             segmented_vns_otsu = otsu_multilevel_thresholding(image, len(vns_thresholds) + 1)[0]
# #             save_image(f'output/{filename}_vns_otsu_k{k}.png', segmented_vns_otsu)

# # if __name__ == "__main__":
# #     main()


# ! latest code with correct naming scheme and doesnt have objective functions
# import cv2
# import numpy as np
# import os

# # Function to load all images from the current folder
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         if filename.endswith(".jpg"):  # Only process .jpg files
#             img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
#             if img is not None:
#                 images.append((filename, img))
#                 print(f"Loaded image: {filename}")  # Debug: Print the image loaded
#             else:
#                 print(f"Failed to load image: {filename}")  # Debug: Failed loading
#     return images

# # Function to create necessary output folders with better naming
# def create_output_folders():
#     base_folder = 'output'
#     methods = ['otsu', 'kapur']
#     optimizations = ['sa', 'vns']
#     for method in methods:
#         for opt in optimizations:
#             for k in [2, 3, 4, 5]:  # Create separate folders for each k level
#                 folder_path = os.path.join(base_folder, f"{opt}_{method}_k{k}")
#                 if not os.path.exists(folder_path):
#                     os.makedirs(folder_path)
#                     print(f"Created folder: {folder_path}")  # Debug: Print folder creation

# # Otsu’s Method Multilevel Thresholding
# def otsu_multilevel_thresholding(image, levels):
#     thresholds = []
#     current_image = image.copy()

#     for _ in range(levels - 1):
#         thresh_value, _ = cv2.threshold(current_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         thresholds.append(thresh_value)
#         current_image = np.where(current_image >= thresh_value, 0, current_image)

#     segmented_image = np.zeros_like(image)
#     for i, thresh in enumerate(sorted(thresholds)):
#         segmented_image[np.where(image >= thresh)] = (i + 1) * (255 // levels)

#     return segmented_image, thresholds

# # Kapur’s Method Thresholding (maximizes entropy)
# def kapur_threshold(image, levels):
#     hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])
#     hist = hist.astype(np.float32) / hist.sum()

#     def entropy(thresh):
#         prob1 = hist[:thresh].sum()
#         prob2 = hist[thresh:].sum()
#         prob1 = prob1 if prob1 > 0 else 1
#         prob2 = prob2 if prob2 > 0 else 1
#         return -(np.log(prob1) * prob1 + np.log(prob2) * prob2)

#     thresholds = []
#     for _ in range(levels - 1):
#         entropies = [entropy(thresh) for thresh in range(1, 255)]
#         best_thresh = np.argmax(entropies)
#         thresholds.append(best_thresh)
#         image = np.where(image >= best_thresh, 0, image)

#     segmented_image = np.zeros_like(image)
#     for i, thresh in enumerate(sorted(thresholds)):
#         segmented_image[np.where(image >= thresh)] = (i + 1) * (255 // levels)

#     return segmented_image, thresholds

# # Simulated Annealing for Thresholding Optimization
# def simulated_annealing(image, objective_function, levels, initial_temperature=1000, cooling_rate=0.95):
#     def perturb(thresholds):
#         idx = np.random.randint(0, len(thresholds))
#         thresholds[idx] += np.random.randint(-10, 10)
#         thresholds = np.clip(thresholds, 0, 255)
#         return thresholds

#     current_thresholds = np.sort(np.random.randint(0, 255, size=(levels - 1)))
#     current_cost = objective_function(image, current_thresholds)
#     temperature = initial_temperature

#     while temperature > 1:
#         new_thresholds = perturb(current_thresholds.copy())
#         new_cost = objective_function(image, new_thresholds)

#         if new_cost < current_cost or np.random.random() < np.exp((current_cost - new_cost) / temperature):
#             current_thresholds = new_thresholds
#             current_cost = new_cost

#         temperature *= cooling_rate

#     return current_thresholds

# # Variable Neighbourhood Search for Thresholding Optimization
# def vns(image, objective_function, levels):
#     current_thresholds = np.sort(np.random.randint(0, 255, size=(levels - 1)))
#     best_cost = objective_function(image, current_thresholds)

#     def local_search(thresholds):
#         for i in range(len(thresholds)):
#             for change in [-5, 5]:
#                 new_thresholds = thresholds.copy()
#                 new_thresholds[i] += change
#                 new_thresholds = np.clip(new_thresholds, 0, 255)
#                 new_cost = objective_function(image, new_thresholds)
#                 if new_cost < best_cost:
#                     return new_thresholds, new_cost
#         return thresholds, best_cost

#     while True:
#         new_thresholds, new_cost = local_search(current_thresholds)
#         if new_cost < best_cost:
#             current_thresholds = new_thresholds
#             best_cost = new_cost
#         else:
#             break

#     return current_thresholds

# # Save segmented image
# def save_image(filename, image):
#     print(f"Saving image to: {filename}")  # Debug: Print saving path
#     cv2.imwrite(filename, image)

# # Main code to load images and apply methods
# def main():
#     folder = './Ass2'  # This points to the current folder where your images are located
#     images = load_images_from_folder(folder)

#     # Check if images are loaded
#     if not images:
#         print("No images found in the folder.")  # Debug: No images case
#         return

#     # Create output folders with better names
#     create_output_folders()

#     for filename, image in images:
#         # Get the filename without the extension
#         name_without_extension = os.path.splitext(filename)[0]
#         print(f"Processing image: {filename}")  # Debug: Print processing image

#         for k in [2, 3, 4, 5]:
#             print(f"Processing k={k}")  # Debug: Print current threshold level

#             # Simulated Annealing + Otsu
#             sa_thresholds = simulated_annealing(image, otsu_objective, k)
#             segmented_sa_otsu = otsu_multilevel_thresholding(image, len(sa_thresholds) + 1)[0]
#             save_image(f'output/sa_otsu_k{k}/{name_without_extension}_sa_otsu_k{k}.png', segmented_sa_otsu)

#             # Simulated Annealing + Kapur
#             sa_thresholds = simulated_annealing(image, kapur_objective, k)
#             segmented_sa_kapur = kapur_threshold(image, len(sa_thresholds) + 1)[0]
#             save_image(f'output/sa_kapur_k{k}/{name_without_extension}_sa_kapur_k{k}.png', segmented_sa_kapur)

#             # Variable Neighbourhood Search + Otsu
#             vns_thresholds = vns(image, otsu_objective, k)
#             segmented_vns_otsu = otsu_multilevel_thresholding(image, len(vns_thresholds) + 1)[0]
#             save_image(f'output/vns_otsu_k{k}/{name_without_extension}_vns_otsu_k{k}.png', segmented_vns_otsu)

#             # Variable Neighbourhood Search + Kapur
#             vns_thresholds = vns(image, kapur_objective, k)
#             segmented_vns_kapur = kapur_threshold(image, len(vns_thresholds) + 1)[0]
#             save_image(f'output/vns_kapur_k{k}/{name_without_extension}_vns_kapur_k{k}.png', segmented_vns_kapur)

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import os

# Function to load all images from the current folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):  # Only process .jpg files
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((filename, img))
                print(f"Loaded image: {filename}")  # Debug: Print the image loaded
            else:
                print(f"Failed to load image: {filename}")  # Debug: Failed loading
    return images

# Function to create necessary output folders with better naming
def create_output_folders():
    base_folder = 'output'
    methods = ['otsu', 'kapur']
    optimizations = ['sa', 'vns']
    for method in methods:
        for opt in optimizations:
            for k in [2, 3, 4, 5]:  # Create separate folders for each k level
                folder_path = os.path.join(base_folder, f"{opt}_{method}_k{k}")
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    print(f"Created folder: {folder_path}")  # Debug: Print folder creation

# Otsu’s Method Multilevel Thresholding
def otsu_multilevel_thresholding(image, levels):
    thresholds = []
    current_image = image.copy()

    for _ in range(levels - 1):
        thresh_value, _ = cv2.threshold(current_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholds.append(thresh_value)
        current_image = np.where(current_image >= thresh_value, 0, current_image)

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
        if len(region) > 0:
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
        entropy = -np.sum
