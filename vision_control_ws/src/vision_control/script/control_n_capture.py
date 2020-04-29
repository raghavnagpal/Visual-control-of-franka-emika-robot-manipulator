#! /usr/bin/env python
"""Data capture node for the Deep CNN network"""

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
import csv
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float64

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
def generate(frame,joint_data):
    global i
    i += 1
    # frame = np.frombuffer(img_data.data, dtype=np.uint8).reshape(img_data.height, img_data.width, -1)
    # cv.namedWindow("win",cv.WINDOW_AUTOSIZE)
    cv.imwrite('/home/taira/work/ws_vision_control/joint_dataset/img%s.jpg'%i,frame)
    # jointValues = data.position
    # q1 = jointValues[0]
    # q2 = jointValues[1]
    with open('/home/taira/work/ws_vision_control/joint_dataset/joint%s.csv'%i, 'w') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_NONE)
        writer.writerow(joint_data)
        csvFile.close()
    print('Generating datsets sample -- ',i)
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
def data_capture(data):
    global steps, t_stamp, running_error, running_joints_val, depth_image_data,value_to_move
    global start_flag

    if start_flag:
        value_to_move = [0.0, 0.0, 0.0, 0.0, 0.0, 1.66, 0.0]
        moveJoint(value_to_move)
        time.sleep(0.2)
        t_stamp = rospy.Time.now()
        start_flag = 0

    if depth_image_data == None or depth_image_data.header.stamp <= t_stamp:
        return

    if data.header.stamp <= t_stamp:
        # print("time is less")
        return
    if running_error > 0.08:
        print("error is high",running_error)
        moveJoint(value_to_move)
        time.sleep(0.2)
        t_stamp = rospy.Time.now()
        return

    # max_angle= 1.22173
    max_angle = 1.0
    samples = 20

    frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

    # img =  iimage.fromarray(frame)
    # img.show()
    frame = frame[:, :, [2, 1, 0]]
    cv.imshow("win", frame)
    cv.waitKey(1)


    framed = np.frombuffer(depth_image_data.data, dtype=np.float32).reshape(depth_image_data.height, depth_image_data.width)
    #
    # img =  iimage.fromarray(frame)
    # img.show()
    framed = framed / 3.0
    framed = framed * 255
    framed = framed.astype(np.uint8)

    cv.imshow("wind", framed)
    cv.waitKey(1)

    generate(framed, running_joints_val)

    joint1_ind = int(steps % samples)
    joint2_ind = int((steps // samples) % samples)
    # file_name = "image_%03d_%03d.png" % (joint1_ind, joint2_ind)
    # cv.imwrite("output/" + file_name, frame)

    # file_name = "image_%03d_%03d.png" % (joint1_ind, joint2_ind)
    # cv.imwrite("output/" + file_name, framed)

    joint1_val = (((max_angle * 2) / samples) * (steps % samples) - max_angle) * (1 - 2 * ((steps // samples) % 2))
    joint2_val = ((max_angle * 2) / samples) * ((steps // samples) % samples) - max_angle

    value_to_move = [joint1_val, joint2_val, 0.0, 0.0, 0.0, 1.66, 0.0]
    moveJoint(value_to_move)
    steps += 1

    time.sleep(0.2)
    # if first:
    #     time.sleep(5)
    t_stamp = rospy.Time.now()

    return


def run(data):
    frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    tstart = time.time()

    height = frame.shape[0]
    width = frame.shape[1]
    frame1 = cv.resize(frame, (576, 324))
    frame_np = frame1[:, :, [2, 1, 0]]

    # test
    img = Image()
    img.encoding = "bgr8"
    img.height = height
    img.width = width
    # img.step = (width) * sizeof(float)
    img.step = img.width * 8 * 3
    img.is_bigendian = 0
    img.data = np.asarray(frame, np.uint8).tostring()
    raw_video_pub.publish(img)
    print("in")

def state_error(data):
    # print("in data")
    global running_error, running_joints_val, state_error_data, value_to_move
    # err_list = [abs(i) for i in data.error.positions]
    # running_error = sum(err_list)
    running_joints_val = data.actual.positions

    state_error_data = np.array(value_to_move) - np.array(running_joints_val)

    err_list = [abs(i) for i in state_error_data]
    running_error = max(err_list)
    # print(running_error)

def cb_depth_image(data):
    # print("in data")
    global depth_image_data
    depth_image_data = data

    # frame = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
    # # frame = np.nan_to_num(frame)
    # frame = frame / 3.0
    #
    # # frame = frame.astype(np.uint8)
    #
    # # frame = frame[:, :, [2, 1, 0]]
    # cv.imshow("windwin", frame)
    # cv.waitKey(1)



if __name__ == '__main__':
    print("init done")
    while (not rospy.is_shutdown()):
        rospy.Subscriber(name='/camera_link/depth/image_raw', data_class=numpy_msg(Image),
                         callback=cb_depth_image,
                         queue_size=10)

        rospy.Subscriber(name='/camera_link/rgb/image_raw', data_class=numpy_msg(Image),
                         callback=data_capture,
                         queue_size=10)
        rospy.Subscriber(name='/arm_controller/state', data_class=JointTrajectoryControllerState,
                         callback=state_error,
                         queue_size=10)

        rospy.spin()

print datetime.datetime.now()
