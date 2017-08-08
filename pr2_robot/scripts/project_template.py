#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
from perception_functions import *

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl( pcl_msg )

    # Statistical Outlier Filter
    cloud_stat_filtered = statistical_outlier( cloud )

    # Voxel Grid Downsampling
    cloud_vox_filtered = vox_downsample( cloud_stat_filtered )

    # PassThrough Filter
    cloud_pt_filtered = passthrough_filter( cloud_vox_filtered )

    # RANSAC Plane Segmentation
    # TODO: Extract inliers and outliers
    cloud_table, cloud_objects = ransac_segmentation( cloud_pt_filtered )
    

    # Euclidean Clustering
    cluster_indices, white_cloud = euclid_cluster( cloud_objects )

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud = cluster_mask( cluster_indices, white_cloud )

    # Convert PCL data to ROS messages
    ros_cloud_objects =  pcl_to_ros( cloud_objects )
    ros_cloud_table   =  pcl_to_ros(  cloud_table  )
    ros_cluster_cloud =  pcl_to_ros( cluster_cloud )

    # Publish ROS messages
    pcl_objects_pub.publish( ros_cloud_objects )
    pcl_table_pub.publish(   ros_cloud_table   )
    pcl_cluster_pub.publish( ros_cluster_cloud )

# Exercise-3 TODOs:
#
#    # Classify the clusters! (loop through each detected cluster one at a time)
#    detected_objects_labels = []
#    detected_objects = []
#    
#    # Grab the points for the cluster
#    for index, pts_list in enumerate(cluster_indices):
#        # Grab the points for the cluster
#        pcl_cluster = cloud_objects.extract(pts_list)
#        # convert the cluster from pcl to ROS using helper function
#	ros_cluster = pcl_to_ros(pcl_cluster)
#
#        # Extract histogram features
#	chists = compute_color_histograms(ros_cluster, using_hsv=True)
#        normals = get_normals(ros_cluster)
#        nhists = compute_normal_histograms(normals)
#        feature = np.concatenate((chists, nhists))
#
#        # Make the prediction
#        # - retrieve the label for the result
#        # - add it to detected_objects_labels list
#        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
#        label = encoder.inverse_transform(prediction)[0]
#        detected_objects_labels.append(label)
#
#        # Publish a label into RViz
#        label_pos = list(white_cloud[pts_list[0]])
#        label_pos[2] += .4
#        object_markers_pub.publish(make_label(label,label_pos, index))
#
#        # Add the detected object to the list of detected objects.
#        do = DetectedObject()
#        do.label = label
#        do.cloud = ros_cluster
#        detected_objects.append(do)
#
#        # Compute the associated feature vector
#
#
#    # Publish the list of detected objects
#    pcl_objects_pub.publish(ros_cloud_objects)
#    pcl_table_pub.publish(ros_cloud_table)
#    pcl_cluster_pub.publish(ros_cluster_cloud)
#    # This is the output you'll need to complete the upcoming project!
#    detected_objects_pub.publish(detected_objects)
#
#    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
#    # Could add some logic to determine whether or not your object detections are robust
#    # before calling pr2_mover()
#    try:
#        pr2_mover(detected_objects_list)
#    except rospy.ROSInterruptException:
#        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # Get/Read parameters
    #object_list_param = rospy.get_param('/object_list')

    # Parse parameters into individual variables
    #object_name = object_list_param[i]['name']
    #object_group = object_list_param[i]['group']

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber( "/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1 )

    # Create Publishers
    pcl_objects_pub = rospy.Publisher( "/pcl_objects", PointCloud2, queue_size=1 )
    pcl_table_pub   = rospy.Publisher( "/pcl_table",   PointCloud2, queue_size=1 )
    pcl_cluster_pub = rospy.Publisher( "/pcl_cluster", PointCloud2, queue_size=1 )
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
     rospy.spin()
