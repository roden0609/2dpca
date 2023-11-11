import cv2
import time
import numpy as np

from pca import PCA
from two_d_pca import TwoDPCA

from image_matrix import ImageMatrix
from two_d_image_matrix import TwoDImageMatrix
from dataset import Dataset

# algorithm = 2d-pca / pca
# algorithm = "2d-pca"
algorithm = "pca"

# No of images For Training, 8 for ORL, 9 for Yale, 21 for AR (Left will be used as testing Image)
no_of_train_images_of_one_person = 21
# dir = faces/ORL or faces/Yale or faces/AR
# dataset_obj = Dataset("faces/ORL", no_of_train_images_of_one_person)
# dataset_obj = Dataset("faces/Yale", no_of_train_images_of_one_person)
dataset_obj = Dataset("faces/AR", no_of_train_images_of_one_person)

# Data For Training
images_names = dataset_obj.images_name_for_train
y = dataset_obj.y_for_train
no_of_elements = dataset_obj.no_of_elements_for_train
target_names = dataset_obj.target_name_as_array

# Data For Testing
images_names_for_test = dataset_obj.images_name_for_test
y_for_test = dataset_obj.y_for_test
no_of_elements_for_test = dataset_obj.no_of_elements_for_test

training_start_time = time.process_time()
img_width, img_height = 50, 50

if algorithm == "pca":
    image_matrix = ImageMatrix(images_names, img_width, img_height)
else:
    image_matrix = TwoDImageMatrix(images_names, img_width, img_height)

scaled_face = image_matrix.get_matrix()

# if algorithm == "pca":
#     cv2.imshow("Original PCA Face",
#                cv2.resize(np.array(np.reshape(scaled_face[:, 1], [img_height, img_width]), dtype=np.uint8), (200, 200)))
#     cv2.waitKey()
# else:
#     cv2.imshow("Original 2D-PCA Face", cv2.resize(scaled_face[0], (200, 200)))
#     cv2.waitKey()

if algorithm == "pca":
    my_algo = PCA(scaled_face, y, target_names, no_of_elements, 10)
else:
    my_algo = TwoDPCA(scaled_face, y, target_names, 90)

new_coordinates = my_algo.reduce_dim()
# if algorithm == "pca":
#     # change the eig_no to show different eigen face
#     my_algo.show_eigen_face(img_width, img_height, 50, 150, 0)

# if algorithm == "pca":
#     cv2.imshow("After PCA Face", cv2.resize(
#         np.array(np.reshape(my_algo.original_data(new_coordinates[1, :]), [img_height, img_width]), dtype=np.uint8),
#         (200, 200)))
#     cv2.waitKey()
# else:
#     cv2.imshow("After 2D-PCA Face",
#                cv2.resize(np.array(my_algo.original_data(new_coordinates[0]), dtype=np.uint8), (200, 200)))
#     cv2.waitKey()

training_time = time.process_time() - training_start_time

time_start = time.process_time()

correct_ary = []
incorrect_ary = []
time_elapsed_ary = []
training_time_ary = []

for _ in range(10):
    correct = 0
    incorrect = 0
    i = 0
    net_time_of_reco = 0

    for img_path in images_names_for_test:

        time_start = time.process_time()
        find_name = my_algo.recognize_face(my_algo.new_cord(img_path, img_height, img_width))
        time_elapsed = (time.process_time() - time_start)
        net_time_of_reco += time_elapsed
        rec_y = y_for_test[i]
        rec_name = target_names[rec_y]
        if find_name is rec_name:
            correct += 1
            # print("Correct -", "Real Person:", rec_name, ", Find Person:", find_name)
        else:
            incorrect += 1
            # print("Incorrect -", "Real Person:", rec_name, ", Find Person:", find_name)
        i += 1

    # Append results to lists
    correct_ary.append(correct)
    incorrect_ary.append(incorrect)
    time_elapsed_ary.append(time_elapsed)
    training_time_ary.append(training_time)

print("Total Person:", len(target_names))
print("Total Train Faces:", no_of_train_images_of_one_person * len(target_names))
print("Total Tested Faces:", i)

# Calculate means excluding the highest and lowest values
mean_correct = sum(sorted(correct_ary)[1:-1]) / 8
mean_incorrect = sum(sorted(incorrect_ary)[1:-1]) / 8
mean_time_elapsed = sum(sorted(time_elapsed_ary)[1:-1]) / 8
mean_training_time = sum(sorted(training_time_ary)[1:-1]) / 8

print("Mean Correct:", mean_correct)
print("Incorrect:", mean_incorrect)
print("Accuracy (%):", mean_correct / i * 100)
print("Total time taken for recognition (ms):", mean_time_elapsed)
# print("Time Taken for one recognition:", time_elapsed / i)
print("Training time (ms):", mean_training_time)
