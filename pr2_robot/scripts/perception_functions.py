#!/usr/bin/env python

# Import modules
import sklearn
from sklearn.preprocessing import LabelEncoder
from sensor_stick.srv import GetNormals
from sensor_stick.pcl_helper import *
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Voxel Grid Downsampling
def vox_downsample(pcl_cloud):
    vox = pcl_cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    return cloud_filtered
    
# PassThrough Filter
def passthrough_filter(pcl_cloud):
    passthrough = pcl_cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.75
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -3
    axis_max = -1.35
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filteredy = passthrough.filter()
    
    return cloud_filteredy

# RANSAC Plane Segmentation
def ransac_segmentation(pcl_cloud):
    seg = pcl_cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.002
    seg.set_distance_threshold(max_distance)

    # Extract inliers and outliers
    #outlier_filter = pcl_cloud.make_statistical_outlier_filter()
    #outlier_filter.set_mean_k(50)
    #outlier_filter.set_std_dev_mul_thresh(1.0)
    #cloud_filtered = outlier_filter.filter()
    
    inliers, coefficients = seg.segment()
    cloud_table   = pcl_cloud.extract(inliers, negative=False)
    cloud_objects = pcl_cloud.extract(inliers, negative=True )

    return cloud_table, cloud_objects
    
# Euclidean Clustering
def euclid_cluster(pcl_cloud):
    white_cloud = XYZRGB_to_XYZ(pcl_cloud) # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(600)
    ec.set_MaxClusterSize(3000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    return cluster_indices, white_cloud


def cluster_mask(cluster_indices, white_cloud):
    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
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

    return cluster_cloud


def classify_cluster ( cluster_indices, white_cloud, cloud_objects, clf, encoder, scaler, object_markers_pub ):
    detected_objects_labels = []
    detected_objects = []
    
    # Grab the points for the cluster
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
	ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # TODO: complete this step just as you did before in capture_features.py
	chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    return detected_objects, detected_objects_labels
