# Visual-control-of-franka-emika-robot-manipulator
Visual control of franka emika robot manipulator

Note: You may need to install dependancies first. See below.

Steps to run the setup:
1.) catkin_make the workspace
  ```
  cd vision_control_ws/
  catkin_make
  source devel/setup.sh
  ```

2.) Run Gazebo simulator and trajectory controllers
  from directory vision_control_ws/
  ```
  source devel/setup.sh
  roslaunch vision_control gazebo.launch 
  ```
  
  System and version info:
  >Ros Melodic
  >Ubuntu 18.04
  >python2.7
  >tensorflow 1.15.2
