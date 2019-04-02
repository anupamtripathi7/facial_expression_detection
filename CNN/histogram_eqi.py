'''
Apply Histogram Equalization to entire dataset
'''


import cv2
import os

if __name__ == '__main__':
    image_directory = r'C:\Users\Dell\Downloads\ck+\cohn-kanade-images'
    count1, count2 = 0, 0

    for main_folder in os.listdir(image_directory):
        directory1 = image_directory + "\\" + main_folder
        print("Retrieving data from ", main_folder)
        for sub_folder in os.listdir(directory1):
            if sub_folder.startswith('.'):
                pass
            else:
                directory2 = directory1 + "\\" + sub_folder
                print(sub_folder)
                # label_path = label_directory + "\\" + main_folder + "\\" + sub_folder
                # label_path = os.path.join(label_directory, main_folder, sub_folder)
                counter = 0
                for img in os.listdir(directory2):
                    if not img.startswith('.'):
                        counter += 1
                        count1 += 1
                        path = os.path.join(directory2, img)
                        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        cv2.equalizeHist(image)
                        cv2.imwrite(path, image)


