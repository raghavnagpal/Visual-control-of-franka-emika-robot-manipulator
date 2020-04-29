import os
import numpy as np
import cv2 as cv


DIR = "data"
dir_list  = os.listdir(DIR)

dir_list = [i for i in dir_list if i[-4:] in ['.jpg']]
dir_list.sort()
print(dir_list)

np_img = []
np_gt = []

for i in range(1000):
    img_name = "img"+str(i)+".jpg"
    csv_name = "joint"+str(i)+".csv"

    img = cv.imread(os.path.join(DIR,img_name))
    if img is None:
        continue

    csv = np.genfromtxt(os.path.join(DIR, csv_name), delimiter=",")

    img = cv.resize(img,(200,200))

    cv.imshow("img",img)
    print(csv)

    cv.waitKey(1)

    np_img.append((img/255.0)*2.0 - 1.0)
    np_gt.append(csv/3.14)


img_array = np.asarray(np_img)
gt_tx = np.asarray(np_gt)




np.savez('dataset_3d_train.npz', img=img_array, gt_tx=gt_tx)

print("dataset saved")