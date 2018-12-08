#!/usr/env/python

import cv2
import numpy as np
import collections
# from mvnc import mvncapi as mvnc


def inverse_kinematics(prediction):
    # Distance between the wheels
    baseline = 0.102

    original_v = 0.386400014162
    original_omega = prediction

    vel = 0.25
    omega = original_omega * vel / original_v


    # assuming same motor constants k for both motors
    k_r = 27.0
    k_l = 27.0
    gain = 1.0
    trim = 0.0
    radius = 0.0318

    # adjusting k by gain and trim
    k_r_inv = (gain + trim) / k_r
    k_l_inv = (gain - trim) / k_l

    omega_r = (vel + 0.5 * omega * baseline) / radius
    omega_l = (vel - 0.5 * omega * baseline) / radius

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv
    return [u_l, u_r]


def fun_img_preprocessing(image, image_final_height, image_final_width):

    # crop the 1/3 upper part of the image
    new_img = image[image.shape[0]/3:, :, :]

    # transform the color image to grayscale
    # new_img = cv2.cvtColor(new_img[:, :, :], cv2.COLOR_RGB2GRAY)

    # resize the image from 320x640 to 48x96
    new_img = cv2.resize(new_img, (image_final_width, image_final_height))

    # normalize images to range [0, 1] (divide each pixel by 255)
    # first transform the array of int to array of float else the division with 255 will return an array of 0s
    new_img = new_img.astype(float)
    new_img = new_img / 255

    # new_part
    new_img = np.reshape(new_img, (1, -1))

    return new_img



class CNN_image_stack:

    def __init__(self,bs,dp):
        self.num_of_backsteps = bs
        self.dropout = dp

        self.img_height = 48
        self.img_width = 96
        self.img_channels = 3
        self.img_flatten_size = self.img_height * self.img_width * self.img_channels

        self.full_img_stack_len = self.num_of_backsteps * (self.dropout - 1)
        self.full_img_stack = collections.deque(self.full_img_stack_len * [self.img_flatten_size * [0]],
                                                self.full_img_stack_len)
        self.img_stack = []

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

        self.img_stack = np.reshape(self.img_stack,(1,-1))

# def load_movidius_graph(path_to_graph):
#
#     # find movidius stick devices
#     devices = mvnc.enumerate_devices()
#     if len(devices) == 0:
#         print('No devices found')
#         quit()
#
#     # get movidius stick device
#     device = mvnc.Device(devices[0])
#
#     # open movidius stick
#     device.open()
#
#     # Load graph
#     with open(path_to_graph, mode='rb') as f:
#         graphFileBuff = f.read()
#
#     graph = mvnc.Graph(path_to_graph)
#     fifoIn, fifoOut = graph.allocate_with_fifos(device, graphFileBuff)
#
#     return graph, fifoIn, fifoOut
#
#
# def movidius_cnn_predictions(graph, fifoIn, fifoOut, img):
#
#     # run CNN
#     graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, img.astype(np.float32), 'user object')
#     output, userobj = fifoOut.read_elem()
#
#     return output[0]
#
#
# def destroy_all(object):
#
#     # close fifo queues, graph and device
#     object.fifoIn.destroy()
#     object.fifoOut.destroy()
#     object.graph.destroy()
#     object.device.close()
