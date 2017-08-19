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
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import pcl
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
    # Extract inliers and outliers
    cloud_table, cloud_objects = ransac_segmentation( cloud_pt_filtered )
    

    # Euclidean Clustering
    cluster_indices, white_cloud = euclid_cluster( cloud_objects )

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    #cluster_cloud = cluster_mask( cluster_indices, white_cloud )
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([
                                            white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float( cluster_color[j] )
                                           ])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)


    # Convert PCL data to ROS messages
    ros_cloud_objects =  pcl_to_ros( cloud_objects )
    ros_cloud_table   =  pcl_to_ros(  cloud_table  )
    ros_cluster_cloud =  pcl_to_ros( cluster_cloud )

    # Publish ROS messages
    pcl_objects_pub.publish( ros_cloud_objects )
    pcl_table_pub.publish(   ros_cloud_table   )
    pcl_cluster_pub.publish( ros_cluster_cloud )

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    
    # Grab the points for the cluster
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        # convert the cluster from pcl to ROS using helper function
	ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
	chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        # - retrieve the label for the result
        # - add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

        # Compute the associated feature vector

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))


    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Compute the centroid of the detected objects
    labels = []
    centroids = [] # to be list of tuples (x, y, z)

    for object in detected_objects:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])

    detected_objects_list = dict(zip(labels, centroids))

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        rospy.loginfo('starting pr2_mover')
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    rospy.loginfo('started pr2_mover')

    # Initialize variables
    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    pick_pose_point = Point()
    place_pose_point = Point()
    place_pose = Pose()
    dropbox_positions = {}
    dropbox_arm_choice = {}
    dropbox = {}
    dict_list = []
    current_picklist = 1

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # sample of dropbox_param:
    # [{'position': [0, 0.71, 0.605], 'group': 'red', 'name': 'left'}, {'position': [0, -0.71, 0.605], 'group': 'green', 'name': 'right'}]

    # Parse parameters into individual variables
    #for dropbox in dropbox_param:
    for entry in dropbox_param:
        if not entry['group'] in dropbox:
            dropbox[entry['group']] = {}
        if not 'position' in dropbox[entry['group']]:
            dropbox[entry['group']]['position'] = {}
        dropbox[entry['group']]['position'] = entry['position']
        if not 'arm_side' in dropbox[entry['group']]:
            dropbox[entry['group']]['arm_side'] = {}
        dropbox[entry['group']]['arm_side'] = entry['name']

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    for object in object_list_param:

        position = object_list.get(object['name'])
	
	if position is not None:

             # Get the PointCloud for a given object and obtain it's centroid
             test_scene_num.data = current_picklist
	     object_name.data = object['name']
             pick_pose_point.x = np.asscalar(position[0])
             pick_pose_point.y = np.asscalar(position[1])
             pick_pose_point.z = np.asscalar(position[2])
             pick_pose.position = pick_pose_point

             # Create 'place_pose' for the object
             place_pose_point.x = dropbox[object['group']]['position'][0]
             place_pose_point.y = dropbox[object['group']]['position'][1]
             place_pose_point.z = dropbox[object['group']]['position'][2]
             place_pose.position = place_pose_point

             # Assign the arm to be used for pick_place
	     arm_name.data = dropbox[object['group']]['arm_side']

             # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
             yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
             dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        #rospy.wait_for_service('pick_place_routine')

        #try:
        #    pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
    #        resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

     #       print ("Response: ",resp.success)

      #  except rospy.ServiceException, e:
       #     print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    yaml_filename = "output_" + str(test_scene_num.data) + ".yaml"
    send_to_yaml(yaml_filename, dict_list)



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

    # Load Model From disk
    model = pickle.load(open('/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
     rospy.spin()
