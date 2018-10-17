#!/usr/bin/env python

import cv2
import os 

import rospy
import cv_bridge
import sensor_msgs.msg
import duckietown_msgs.msg

from mvnc import mvncapi as mvnc
from cnn_lane_following.cnn_predictions import *

class CNN_lane_following:

    def __init__(self, graph_path):

        self.graph, self.fifoIn, self.fifoOut = load_movidius_graph(graph_path)
        rospy.loginfo("graph loaded")

        self.cnn = movidius_cnn_predictions
        self.cvbridge = cv_bridge.CvBridge()

        # Publications
        self.pub = rospy.Publisher("~car_cmd", duckietown_msgs.msg.Twist2DStamped, queue_size=1)

        # Subscriptions
        self.subs = rospy.Subscriber("~compressed", sensor_msgs.msg.CompressedImage, self.receive_img, queue_size=1)

        # execute in case of CTR + C to terminate lane following demo
        rospy.on_shutdown(self.custom_shutdown)

    def receive_img(self, img_msg):

        img = self.cvbridge.compressed_imgmsg_to_cv2(img_msg)
        img = fun_img_preprocessing(img, 48, 96)  # returns image of shape [1, img_height_size x img_width_size]
        prediction = self.cnn(self.graph, self.fifoIn, self.fifoOut, img)

        car_control_msg = duckietown_msgs.msg.Twist2DStamped()
        car_control_msg.header = img_msg.header

        # adjust translation velocity to v=0.25 m/s for smoother driving
        original_v = 0.386400014162
        original_omega = prediction

        new_v = 0.25
        new_omega = original_omega * new_v / original_v

        car_control_msg.v = new_v
        car_control_msg.omega = new_omega

        self.pub.publish(car_control_msg)
        rospy.loginfo("Publishing car_cmd: u={} w={}".format(car_control_msg.v, car_control_msg.omega) )

    def custom_shutdown(self):

        rospy.loginfo("Shutting down!!!")

        car_control_msg = duckietown_msgs.msg.Twist2DStamped()
        car_control_msg.v = 0
        car_control_msg.omega = 0
        self.pub.publish(car_control_msg)
        rospy.loginfo("Publishing car_cmd: u={} w={}".format(car_control_msg.v, car_control_msg.omega))

        # close fifo queues, graph and device
        destroy_all(self)


def main():

    rospy.init_node("cnn_node")
    CNN_lane_following(rospy.get_param("~graph_path"))
    rospy.spin()

if __name__ == "__main__":
    main()
