#!/usr/bin/env python
import rospy
import rospkg
from micromvp_test.msg import micromvp_carspeed
from dagu_car.dagu_wheels_driver import DaguWheelsDriver

class Subspeed_node(object):
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))

		# Open driver
		self.driver = DaguWheelsDriver()

		# Setup subscribers
		self.control_constant = 1.0
		self.sub_topic = rospy.Subscriber("micromvp", micromvp_carspeed, self.cbWheelsCmd, queue_size=1)

		self.max_vel = 200

	def cbWheelsCmd(self,msg):
		
		lspeed = msg.lspeed * self.max_vel
		rspeed = msg.rspeed * self.max_vel #change normalize to real pwm
		self.driver.setWheelsSpeed(left=lspeed,right=rspeed)
		
		# Put the wheel commands in a message and publish
		#self.msg_wheels_cmd.header = msg.header
		# Record the time the command was given to the wheels_driver
		#self.msg_wheels_cmd.header.stamp = rospy.get_rostime()  
		#self.msg_wheels_cmd.vel_left = msg.vel_left
		#self.msg_wheels_cmd.vel_right = msg.vel_right
		#self.pub_wheels_cmd.publish(self.msg_wheels_cmd)

	def on_shutdown(self):
		self.driver.setWheelsSpeed(left=0.0,right=0.0)
		rospy.loginfo("[%s] Shutting down."%(rospy.get_name()))

if __name__ == '__main__':
	# Initialize the node with rospy
	rospy.init_node('subspeed_node', anonymous=False)
	# Create the DaguCar object
	node = Subspeed_node()
	# Setup proper shutdown behavior 
	rospy.on_shutdown(node.on_shutdown)
	# Keep it spinning to keep the node alive
	rospy.spin()