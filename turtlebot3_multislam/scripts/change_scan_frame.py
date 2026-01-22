#!/usr/bin/python3
import rospy
from sensor_msgs.msg import LaserScan

ns = rospy.get_namespace().strip("/")
rospy.init_node("change_scan_frame")
pub = rospy.Publisher("scan_fixed", LaserScan, queue_size=50)

def scan_callback(msg: LaserScan):
    msg.header.frame_id = "base_scan"
    pub.publish(msg)

sub = rospy.Subscriber("scan", LaserScan, scan_callback)

rospy.spin()