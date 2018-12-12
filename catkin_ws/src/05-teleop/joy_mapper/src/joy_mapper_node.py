#!/usr/bin/env python
import rospy
import math

from duckietown_msgs.msg import Twist2DStamped, BoolStamped
from sensor_msgs.msg import Joy

from __builtin__ import True

class JoyMapper(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))

        self.joy = None
        self.last_pub_msg = None
        self.last_pub_time = rospy.Time.now()

        # Setup Parameters
        self.v_gain = self.setupParam("~speed_gain", 0.41)
        self.omega_gain = self.setupParam("~steer_gain", 8.3)
        self.bicycle_kinematics = self.setupParam("~bicycle_kinematics", 0)
        self.steer_angle_gain = self.setupParam("~steer_angle_gain", 1)
        self.simulated_vehicle_length = self.setupParam("~simulated_vehicle_length", 0.18)
        self.alpha_v = self.setupParam("~alpha_v", 0.9)
        self.alpha_omega = self.setupParam("~alpha_omega", 0.2)

        # Publications
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
        self.pub_joy_override = rospy.Publisher("~joystick_override", BoolStamped, queue_size=1)
        self.pub_parallel_autonomy = rospy.Publisher("~parallel_autonomy",BoolStamped, queue_size=1)
        self.pub_anti_instagram = rospy.Publisher("anti_instagram_node/click",BoolStamped, queue_size=1)
        self.pub_e_stop = rospy.Publisher("wheels_driver_node/emergency_stop",BoolStamped,queue_size=1)
        self.pub_avoidance = rospy.Publisher("~start_avoidance",BoolStamped,queue_size=1)

        # Subscriptions
        self.sub_joy_ = rospy.Subscriber("joy", Joy, self.cbJoy, queue_size=1)

        # timer
        self.pub_timer = rospy.Timer(rospy.Duration.from_sec(0.01),self.publishControl)
        self.param_timer = rospy.Timer(rospy.Duration.from_sec(1.0),self.cbParamTimer)
        self.has_complained = False

        self.state_parallel_autonomy = False
        self.deep_learning = False
        self.state_verbose = False

        self.v_state = 0.0
        self.omega_state = 0.0

        self.v_input = 0.0
        self.omega_input = 0.0

        pub_msg = BoolStamped()
        pub_msg.data = self.state_parallel_autonomy
        pub_msg.header.stamp = self.last_pub_time
        self.pub_parallel_autonomy.publish(pub_msg)

    def cbParamTimer(self,event):
        self.v_gain = rospy.get_param("~speed_gain", 1.0)
        self.omega_gain = rospy.get_param("~steer_gain", 10)
        self.alpha_v = rospy.get_param("~alpha_v", 0.9)
        self.alpha_omega = rospy.get_param("~alpha_omega", 0.2)

    def setupParam(self,param_name,default_value):
        value = rospy.get_param(param_name,default_value)
        rospy.set_param(param_name,value) #Write to parameter server for transparancy
        rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
        return value

    def cbJoy(self, joy_msg):
        self.joy = joy_msg
        self.v_input = self.joy.axes[1]
        self.omega_input = self.joy.axes[3]
        # self.publishControl()
        self.processButtons(joy_msg)

    def publishControl(self,event):
        car_cmd_msg = Twist2DStamped()
        # car_cmd_msg.header.stamp = self.joy.header.stamp
        self.v_state = self.v_state * self.alpha_v + (1 - self.alpha_v) * self.v_input * self.v_gain #Left stick V-axis. Up is positive
        if self.v_state < 0.01:
            self.v_state = 0.0
        car_cmd_msg.v = self.v_state
        if self.bicycle_kinematics:
            # Implements Bicycle Kinematics - Nonholonomic Kinematics
            # see https://inst.eecs.berkeley.edu/~ee192/sp13/pdf/steer-control.pdf
            steering_angle = self.joy.axes[3] * self.steer_angle_gain
            car_cmd_msg.omega = car_cmd_msg.v / self.simulated_vehicle_length * math.tan(steering_angle)
        else:
            # Holonomic Kinematics for Normal Driving
            self.omega_state = self.omega_state * self.alpha_omega + (1 - self.alpha_omega) * self.omega_input * self.omega_gain
            if self.omega_state < 0.01:
                self.omega_state = 0.0
            car_cmd_msg.omega = self.omega_state
        self.pub_car_cmd.publish(car_cmd_msg)

# Button List index of joy.buttons array:
# 0: A
# 1: B
# 2: X
# 3: Y
# 4: Left Back
# 5: Right Back
# 6: Back
# 7: Start
# 8: Logitek
# 9: Left joystick
# 10: Right joystick

# XXX: here we should use constants
    def processButtons(self, joy_msg):
        # Button A
        if (joy_msg.buttons[0] == 1):
            self.deep_learning ^= True
            deep_lane_following_msg = BoolStamped()
            rospy.loginfo('start deep learning lane following')
            deep_lane_following_msg.header.stamp = self.joy.header.stamp
            deep_lane_following_msg.data = self.deep_learning
            self.pub_deep_lane_following.publish(deep_lane_following_msg)

        # Y button
        elif (joy_msg.buttons[3] == 1):
            anti_instagram_msg = BoolStamped()
            anti_instagram_msg.header.stamp = self.joy.header.stamp
            anti_instagram_msg.data = True
            rospy.loginfo('anti_instagram message')
            self.pub_anti_instagram.publish(anti_instagram_msg)

        # Left back button
        elif (joy_msg.buttons[4] == 1):
            self.state_parallel_autonomy ^= True
            rospy.loginfo('state_parallel_autonomy = %s' % self.state_parallel_autonomy)
            parallel_autonomy_msg = BoolStamped()
            parallel_autonomy_msg.header.stamp = self.joy.header.stamp
            parallel_autonomy_msg.data = self.state_parallel_autonomy
            self.pub_parallel_autonomy.publish(parallel_autonomy_msg)

        # Right back button
        elif (joy_msg.buttons[5] == 1):
            self.state_verbose ^= True
            rospy.loginfo('state_verbose = %s' % self.state_verbose)
            rospy.set_param('line_detector_node/verbose', self.state_verbose) # bad - should be published for all to hear - not set a specific param

        # Back button
        elif (joy_msg.buttons[6] == 1):
            override_msg = BoolStamped()
            override_msg.header.stamp = self.joy.header.stamp
            override_msg.data = True
            rospy.loginfo('override_msg = True')
            self.pub_joy_override.publish(override_msg)

        # Start button
        elif (joy_msg.buttons[7] == 1):
            override_msg = BoolStamped()
            override_msg.header.stamp = self.joy.header.stamp
            override_msg.data = False
            rospy.loginfo('override_msg = False')
            self.pub_joy_override.publish(override_msg)

        # Power/middle button
        elif (joy_msg.buttons[8] == 1):
            e_stop_msg = BoolStamped()
            e_stop_msg.header.stamp = self.joy.header.stamp
            e_stop_msg.data = True # note that this is toggle (actual value doesn't matter)
            rospy.loginfo('E-stop message')
            self.pub_e_stop.publish(e_stop_msg)

        # Left joystick button
        elif (joy_msg.buttons[9] == 1):
            avoidance_msg = BoolStamped()
            rospy.loginfo('start lane following with avoidance mode')
            avoidance_msg.header.stamp = self.joy.header.stamp
            avoidance_msg.data = True
            self.pub_avoidance.publish(avoidance_msg)

        # # Unknown joystick button
        # elif (joy_msg.buttons[10] == 1):
        #     avoidance_msg = BoolStamped()
        #     rospy.loginfo('executing takeover')
        #     avoidance_msg.header.stamp = self.joy.header.stamp
        #     avoidance_msg.data = True
        #     self.pub_avoidance.publish(avoidance_msg)

        else:
            some_active = sum(joy_msg.buttons) > 0
            if some_active:
                rospy.loginfo('No binding for joy_msg.buttons = %s' % str(joy_msg.buttons))


if __name__ == "__main__":
    rospy.init_node("joy_mapper",anonymous=False)
    joy_mapper = JoyMapper()
    rospy.spin()
