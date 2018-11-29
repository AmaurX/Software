#!/usr/bin/env python

import cv2
import os
import collections
import numpy as np
import tensorflow as tf

import rospy
import cv_bridge
import sensor_msgs.msg
import duckietown_msgs.msg

from cnn_lane_following.cnn_predictions import fun_img_preprocessing,CNN_image_stack
from cnn_lane_following.graph_utils import load_graph



class CNN_lane_following:
    def __init__(self, graph_path, use_bl):

        # We use our "load_graph" function
        self.graph = load_graph(graph_path)

        self.session = tf.Session(graph=self.graph)

        # We can verify that we can access the list of operations in the graph
        for op in self.graph.get_operations():
            print(op.name)
            # prefix/Placeholder/inputs_placeholder
            # ...
            # prefix/Accuracy/predictions

        # We access the input and output nodes
        self.x = self.graph.get_tensor_by_name('prefix/x:0')
        self.y = self.graph.get_tensor_by_name('prefix/ConvNet/fc_layer_out/BiasAdd:0')

        rospy.loginfo("graph loaded")

        self.cvbridge = cv_bridge.CvBridge()

        self.use_bl = use_bl
        self.num_of_backsteps = 5
        self.dropout = 7

        self.img_height = 48
        self.img_width = 96
        self.img_channels = 3

        self.counter = 0
        self.active = False

        self.cnn_image_stack = CNN_image_stack()

        # Publications
        self.pub = rospy.Publisher("~car_cmd", duckietown_msgs.msg.Twist2DStamped, queue_size=1)

        # Subscriptions
        self.subs = rospy.Subscriber("~compressed", sensor_msgs.msg.CompressedImage, self.receive_img, queue_size=1)

        # execute in case of CTR + C to terminate lane following demo
        rospy.on_shutdown(self.custom_shutdown)

    def receive_img(self, img_msg):

        img = self.cvbridge.compressed_imgmsg_to_cv2(img_msg)
        img = fun_img_preprocessing(img, self.img_height, self.img_width)  # returns image of shape [1, img_height_size x img_width_size]

        if self.use_bl:
            self.cnn_image_stack.add_to_stack(img)
            self.counter = (self.counter + 1) % self.cnn_image_stack.full_img_stack_len
            if self.counter == 0:
                self.active = True
                rospy.loginfo("Obtained enough images to (re)start driving")
        else:
            self.img_stack = img

        if self.active:
            output = self.session.run(self.y, feed_dict={
                self.x: self.cnn_image_stack.img_stack
            })

            # print("Output: {}".format(output))

            prediction = output[0][0][0]
        else:
            prediction = 0
        # print("Prediction: {}".format(prediction))

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

    def add_to_stack(self, img):
        """

        :param img: image of shape [1, img_height_size x img_width_size x num_of_channels]
        """

        # self.full_img_stack is of shape [num_of_backsteps x dropout, img_height_size x img_width_size x num_of_channels]
        self.full_img_stack.appendleft(img)

        # self.img_stack is of shape [1, num_of_backsteps x img_height_size x img_width_size x num_of_channels]
        self.img_stack = []
        for i in range(0, self.full_img_stack_len, self.dropout):
            self.img_stack = np.append(self.img_stack, self.full_img_stack[i])

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
    CNN_lane_following(rospy.get_param("~graph_path"),rospy.get_param("~use_bl"))
    rospy.spin()

if __name__ == "__main__":
    main()
