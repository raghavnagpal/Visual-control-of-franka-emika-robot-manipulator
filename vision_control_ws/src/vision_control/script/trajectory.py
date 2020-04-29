#! /usr/bin/env python
"""Publishes joint trajectory to move robot to given trajectory"""

import rospy
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_srvs.srv import Empty
import argparse
import glob
import datetime
import time
import numpy as np
import cv2 as cv
import os
import csv
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float64
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.backend import set_session
# tensorflow 1.15.0

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

rospy.init_node('data_capture_node', anonymous=True)
joint_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
raw_video_pub = rospy.Publisher('RAW_Video_pub1', Image, queue_size=1)

steps = 0

first = True
# use iterator later
# range -1.22173 to 1.22173, steps
t_stamp = rospy.Time.now()
running_error = 10.0
running_joints_val = []
depth_image_data = None
state_error_data = None
value_to_move = [0.0, 0.0, 0.0, 0.0, 0.0, 1.66, 0.0]

i = 0


def generate(frame, joint_data):
    global i
    i += 1
    # frame = np.frombuffer(img_data.data, dtype=np.uint8).reshape(img_data.height, img_data.width, -1)
    # cv.namedWindow("win",cv.WINDOW_AUTOSIZE)
    cv.imwrite('/home/taira/work/ws_vision_control/joint_dataset/img%s.jpg' % i, frame)
    # jointValues = data.position
    # q1 = jointValues[0]
    # q2 = jointValues[1]
    with open('/home/taira/work/ws_vision_control/joint_dataset/joint%s.csv' % i, 'w') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_NONE)
        writer.writerow(joint_data)
        csvFile.close()
    print('Generating datsets -- ', i)
    time.sleep(0.20)


def moveJoint(jointcmds, prefix='panda', nbJoints=7):
    jointCmd = JointTrajectory()
    point = JointTrajectoryPoint()
    jointCmd.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(0.0);
    point.time_from_start = rospy.Duration.from_sec(5.0)
    for i in range(0, nbJoints):
        jointCmd.joint_names.append(prefix + '_joint' + str(i + 1))
        point.positions.append(jointcmds[i])
        point.velocities.append(0)
        point.accelerations.append(0)
        point.effort.append(0)
    jointCmd.points.append(point)
    rate = rospy.Rate(100)
    count = 0
    while (count < 5):
        joint_pub.publish(jointCmd)
        count = count + 1
        rate.sleep()
        # print("pubs")


start_flag = 1

joint_samples = []

def make_trajectory():
    global joint_samples

    for i in range(600):
        angle = i * np.pi / 180
        value_to_move = [np.sin(angle), np.cos(angle), 0.0, 0.0, 0.0, 1.66, 0.0]
        joint_samples.append(value_to_move)

    joint_samples = np.array(joint_samples)


def get_trajectory(joint_in=[np.sin(1), np.cos(1), 0.0, 0.0, 0.0, 1.66, 0.0]):
    global joint_samples

    joint_in = np.array(joint_in)
    joints_diff = joint_samples - joint_in
    joints_diff = np.abs(joints_diff)
    joints_err = np.sum(joints_diff,axis=1)
    index = np.argmin(joints_err)
    print("index",index,"running error", joints_err[index])

    if joints_err[index] > 1.0:
        return joint_samples[index+10]
    else:
        return joint_samples[index+20]

    return None


def trajtory(data):
    global steps
    j_pos = data.actual.positions
    new_j_pos = get_trajectory(j_pos)
    moveJoint(new_j_pos)
    # time.sleep(0.1)


class deep_network(object):
    def __init__(self):
        self.model = models.load_model('/home/taira/work/visula_control/model/end_model_1.h5')
        # self.model._make_predict_function()

    def get_joint_from_image(self,frame):
        # graph = tf.get_default_graph()
        img = np.expand_dims(frame,axis=0)
        # with graph.as_default():
        #     output = self.model.predict(img)
        output = self.model.predict(img)

        return output


sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

model = models.load_model('/home/taira/work/visula_control/model/end_model_1.h5')
model._make_predict_function()


def cb_trajectory_deep(data):
    global graph, model, sess

    framed = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
    # img =  iimage.fromarray(frame)
    # img.show()
    framed = framed / 3.0
    framed = framed * 255
    framed = framed.astype(np.uint8)

    cv.imshow("wind", framed)
    cv.waitKey(1)
    framed = cv.resize(framed,(200,200))
    framed = np.stack([framed,framed,framed],axis=2)
    framed = (framed /255.0)*2.0 - 1.0

    with graph.as_default():
        set_session(sess)
        img = np.expand_dims(framed,axis=0)
        j_pos = model.predict(img)

    j_pos = j_pos * 3.14
    new_j_pos = get_trajectory(j_pos[0])
    moveJoint(new_j_pos)

    t_stamp = rospy.Time.now()

    return


def trajtory_test(data):
    global steps
    value_to_move = [np.sin(steps / 10), np.cos(steps / 10), 0.0, 0.0, 0.0, 1.66, 0.0]
    moveJoint(value_to_move)
    steps += 1
    time.sleep(1.0)


if __name__ == '__main__':
    global dl_obj
    print("init done")
    make_trajectory()

    while (not rospy.is_shutdown()):

        rospy.Subscriber(name='/camera_link/depth/image_raw', data_class=numpy_msg(Image),
                                          callback=cb_trajectory_deep,
                                          queue_size=1)

        rospy.spin()


print datetime.datetime.now()
