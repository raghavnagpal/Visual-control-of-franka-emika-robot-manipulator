import numpy as np
import os
from tensorflow.keras import datasets, layers, models
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

counter = 0
input_file = "dataset_3d_train.npz"
# input_file = "dataset_test.npz"

data = np.load(input_file)

# dataset numpy
img_array = data['img']
gt_tx = data['gt_tx']

# Save the model
model = models.load_model('model/end_model_1.h5')
print("model load")

joint_input = []
joint_output = []
joint_error = []
index = []

for i in range(len(img_array)):
    img = img_array[i]
    img = np.expand_dims(img,axis=0)
    output = model.predict(img)
    print("gt = ",gt_tx[i],"detected = ",output[0],"diff = ",gt_tx[i] - output[0])

    joint_input.append(gt_tx[i])
    joint_output.append(output[0])
    joint_error.append(gt_tx[i] - output[0])
    index.append(i)

    cv.imshow("win",img[0])
    cv.waitKey(1)

# example_batch = img_array[:10]
# example_batch_mask =  img_array[:10]
# example_batch_test = gt_tx[:10]
# example_result = model.predict(example_batch_mask)
# print(example_batch_test)
# print(example_result)

joint_input = np.array(joint_input)
joint_output = np.array(joint_output)
joint_error = np.array(joint_error)
index = np.array(index)

fig = plt.figure("Joint values ground truth")
ax = fig.add_subplot(111)
ax.plot(index,joint_input[:,0],index,joint_input[:,1])
ax.set_xlabel('time step')
ax.set_ylabel('Joint values ground truth')
ax.legend(["Joint 2","Joint 5"])

fig = plt.figure("Joint values predicted")
ax = fig.add_subplot(111)
ax.plot(index,joint_output[:,0],index,joint_output[:,1])
ax.set_xlabel('time step')
ax.set_ylabel('Joint values predicted')
ax.legend(["Joint 2","Joint 5"])

fig = plt.figure("Error in joint prediction")
ax = fig.add_subplot(111)
ax.plot(index,joint_error[:,0],index,joint_error[:,1])
ax.set_xlabel('time step')
ax.set_ylabel('error ')
ax.legend(["Joint 2","Joint 5"])

plt.show()

print("end")