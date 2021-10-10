from __future__ import division, print_function

import rospy
import os
import tf
import message_filters
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from trt_yolo.detector import DarknetTRT
from utils import timeit_ros


import Queue as queue


class YOLORos(object):
    def __init__(self):
        self._bridge = CvBridge()
        self._read_params()
        self.model = DarknetTRT(
            obj_threshold=self.obj_threshold,
            nms_threshold=self.nms_threshold,
            yolo_type=self.yolo_type,
            weights_path=self.weights_path,
            config_path=self.config_path,
            label_filename=self.label_filename,
            postprocessor_cfg=self.postprocessor_cfg,
            cuda_device=self.cuda_device,
            show_image=self.publish_image,
            IMAGE_PATH=self.IMAGE_PATH
        )
        self.lidar_sub_switch=False
        self.count_tmp = 1
        self.corresponding_seq = 0
        self.class_dict = {}
        self.landingSign = Bool()
        self.landingSign.data = False
        self.RTBSign = False
        self.num_threshold_to_delete = 2
        self.msg_queue = queue.Queue(maxsize=5)
        self._init_topics()
        rospy.loginfo("[trt_yolo_ros] loaded and ready")
        rospy.loginfo("init yolo")
        
        
    def _read_params(self):
        """ Reading parameters for YOLORos from launch or yaml files """
        self.publish_image = rospy.get_param("~publish_image", False)
        self.print_outcome = rospy.get_param("~print_outcome", False)
        # default paths to weights from different sources
        self.weights_path = rospy.get_param("~weights_path", "./weights/")
        self.config_path = rospy.get_param("~config_path", "./config/")
        self.label_filename = rospy.get_param("~label_filename", "coco_labels.txt")
        # parameters of yolo detector
        self.yolo_type = rospy.get_param("~yolo_type", "yolov3-416")
        self.postprocessor_cfg = rospy.get_param(
            "~postprocessor_cfg", "yolo_postprocess_config.json"
        )
        self.obj_threshold = rospy.get_param("~obj_threshold", 0.6)
        self.nms_threshold = rospy.get_param("~nms_threshold", 0.3)
        # default cuda device
        self.cuda_device = rospy.get_param("~cuda_device", 0)
        self.num_cameras = rospy.get_param("~num_cam", 1)
        self.IMAGE_PATH = rospy.get_param("~IMAGE_PATH", "./IMAGE/")
        self.TEXT_PATH = rospy.get_param("~TEXT_PATH", "./TEXT/")
        self.makeDIR()
        rospy.logdebug("[trt_yolo_rs]: Number of cameras", self.num_cameras)
        rospy.logdebug("[trt_yolo_ros] parameters read")

    @staticmethod
    def _read_subscriber_param(name):
        """ reading subscriber parameters from launch or yaml files """
        topic = rospy.get_param("~subscriber/" + name + "/topic")
        queue_size = rospy.get_param("~subscriber/" + name + "/queue_size", 10)
        return topic, queue_size

    @staticmethod
    def _read_publisher_param(name):
        """ reading publisher parameters from launch or yaml files """
        topic = rospy.get_param("~publisher/" + name + "/topic")
        queue_size = rospy.get_param("~publisher/" + name + "/queue_size", 1)
        latch = rospy.get_param("~publisher/" + name + "/latch", False)
        return topic, queue_size, latch

    def _init_topics(self):
        """ This function is initializing node publisher and subscribers for the node """
        # Publisher
        topic, queue_size, latch = self._read_publisher_param("bounding_boxes")
        self._pub = rospy.Publisher(
            topic, BoundingBoxes, queue_size=queue_size, latch=latch
        )

        topic, queue_size, latch = self._read_publisher_param("image")
        self._pub_viz = rospy.Publisher(
            topic, Image, queue_size=queue_size, latch=latch
        )

        self._pub_object = rospy.Publisher(
            '/detectedObject', PoseStamped, queue_size=10
        )

        self._pub_object_scan = rospy.Publisher(
            '/objectScan', LaserScan, queue_size=10
        )

        self._pub_landing_sign =  rospy.Publisher(
            '/landingSign', Bool, queue_size=1
        )

        self._sub_landing_sign =  rospy.Subscriber(
            '/RTBSign', Bool, self.RTB_callback
        )

        # Image Subscriber
        for i in range(self.num_cameras):
            topic, queue_size = self._read_subscriber_param("image")
            # self._image_sub = rospy.Subscriber(
            #     topic,
            #     Image,
            #     self._image_callback,
            #     queue_size=queue_size,
            #     buff_size=2 ** 24,
            # )
            self._image_sub = message_filters.Subscriber(topic,Image)
        
        rospy.logdebug("[trt_yolo_ros] publishers and subsribers initialized")
        # self.pointcloud_sub = rospy.Subscriber("/camera/depth/color/points",PointCloud2,self._pointcloud_callback,queue_size=20)
        self.lidar_sub = message_filters.Subscriber("/scan",LaserScan)
        self.pose_sub = message_filters.Subscriber("/mavros/local_position/pose", PoseStamped)

        self.landing_sub = rospy.Subscriber("/landingSign", Bool,self.landing_callback)

        self.mf = message_filters.ApproximateTimeSynchronizer([self._image_sub, self.lidar_sub, self.pose_sub], 5,0.1)
        self.mf.registerCallback(self.mf_callback)
        self.lidar_data = LaserScan()
        self._tfpub = tf.TransformBroadcaster()
        self._tfsub = tf.TransformListener()

        self.transMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.1],[0,0,0,1]]) # Instead of static tf, Use transformation using numpy 
        self.drone_pose_map = PoseStamped()
        
    def makeDIR(self):
        image_list = os.listdir(self.IMAGE_PATH)
        text_list = os.listdir(self.TEXT_PATH)

        dir_index = 0

        while str(dir_index) in image_list:
            dir_index = dir_index + 1
        self.IMAGE_PATH = self.IMAGE_PATH+str(dir_index)+"/"
        self.TEXT_PATH = self.TEXT_PATH+str(dir_index)+"/"
        os.mkdir(self.IMAGE_PATH)
        os.mkdir(self.TEXT_PATH)

    def mf_callback(self,image, pointcloud, pose):
        self._lidar_callback(pointcloud)
        self._pose_callback(pose)
        self._image_callback(image)

    def RTB_callback(self, msg):
        if msg.data:
            self.RTBSign = True
            rospy.loginfo("RTB Sign received...")

    def _pose_callback(self, msg):
        self.drone_pose_map = msg
        self.corresponding_seq = msg.header.seq

    def _lidar_callback(self,msg):
        self.lidar_data = msg        
        if self.lidar_sub_switch is False:
            self.lidar_sub_switch = True
        if self.RTBSign: 
            getPositionWhileLanding(msg)


    def _image_callback(self, msg):
        """ Main callback which is saving the last received image """
        self.IMAGE_WIDTH = msg.width
        if msg.header is not None:
            self.msg_queue.put(msg)
            rospy.logdebug("[trt_yolo_ros] image recieved")

    def landing_callback(self, msg):
        if msg.data:
            image_list = os.listdir(self.IMAGE_PATH)
            text_list = os.listdir(self.TEXT_PATH)

            for text_file in text_list:
                f = open(self.TEXT_PATH+text_file,'r+')
                lines = f.readlines()

                if len(lines) < self.num_threshold_to_delete:
                    os.remove(self.TEXT_PATH+text_file)
                    os.remove(self.IMAGE_PATH+text_file[:-4]+".jpg")
                    rospy.loginfo(text_file+" is removed, it has little measurement  -->> Assunm as Case of False Positive")
                else:
                    # For now just calculate the average
                    f_result = open(self.TEXT_PATH+text_file[:-4]+"_final.txt",'w')
                    x_sum = 0
                    y_sum = 0
                    for line in lines:
                        x,y = line.split()
                        x_sum = x_sum + float(x)
                        y_sum = y_sum + float(y)
                    x_avg = x_sum / len(lines)
                    y_avg = y_sum / len(lines)
                    f_result.write(str(x_avg)+" "+str(y_avg))
                    os.remove(self.TEXT_PATH+text_file)
                    rospy.loginfo(text_file+" Result : "+ str(x_avg)+" "+str(y_avg))
                    f_result.close()
                f.close()

    def extractingObjectFromLidar(self, lidar_data,left,right):
        try:
            start_index = int(460+((self.IMAGE_WIDTH-right)//4))
            end_index = int(640-(left//4))
            objectRange = lidar_data.ranges[start_index:end_index]
            object_data = LaserScan()
            object_data.header = lidar_data.header
            object_data.angle_min = (start_index*(np.pi/720))-((3/4)*np.pi)
            object_data.angle_max = (end_index*(np.pi/720))-((3/4)*np.pi)
            object_data.angle_increment = lidar_data.angle_increment
            object_data.time_increment = lidar_data.time_increment
            object_data.scan_time = lidar_data.scan_time
            object_data.range_min = lidar_data.range_min
            object_data.range_max = lidar_data.range_max
            object_data.ranges = objectRange
            object_data.intensities = []
            self._pub_object_scan.publish(object_data)
            rospy.loginfo(str(left) + " " + str(right))

            mid_index = (start_index+end_index)//2
            mid_range = lidar_data.ranges[mid_index]
            mid_angle = (mid_index*(np.pi/720))-((3/4)*np.pi)
            object_x = mid_range*np.cos(mid_angle)
            object_y = mid_range*np.sin(mid_angle)
        except:
            object_x = lidar_data.ranges[len(lidar_data.ranges)//2]
            object_y = 0 
        return [object_x, object_y]

    def extractingObjectFromLidarDog(self, lidar_data,left,right):
        try:
            start_index = int(460+((right-self.IMAGE_WIDTH-right)//4))
            end_index = int(640-(left//4))
            objectRange = lidar_data.ranges[start_index:end_index]
            object_data = LaserScan()
            object_data.header = lidar_data.header
            object_data.angle_min = (start_index*(np.pi/720))-((3/4)*np.pi)
            object_data.angle_max = (end_index*(np.pi/720))-((3/4)*np.pi)
            object_data.angle_increment = lidar_data.angle_increment
            object_data.time_increment = lidar_data.time_increment
            object_data.scan_time = lidar_data.scan_time
            object_data.range_min = lidar_data.range_min
            object_data.range_max = lidar_data.range_max
            object_data.ranges = objectRange
            object_data.intensities = []
            self._pub_object_scan.publish(object_data)

            mid_index = (start_index+end_index)//2
            mid_range = lidar_data.ranges[mid_index]
            mid_angle = (mid_index*(np.pi/720))-((3/4)*np.pi)
            object_x = mid_range*np.cos(mid_angle)/2
            object_y = mid_range*np.sin(mid_angle)
        except:
            object_x = lidar_data.ranges[len(lidar_data.ranges)//2]
            object_y = 0 
        return [object_x, object_y]

    def getPositionWhileLanding(self, lidar_data):
        start_index = 360
        end_index = 720

        objectRange = lidar_data.ranges[start_index:end_index]
        size_of_range = len(objectRange)
        angles = np.linspace((start_index*(np.pi/720))-((3/4)*np.pi),(end_index*(np.pi/720))-((3/4)*np.pi),size_of_range)
        lidar_x = np.array(objectRange)*np.cos(angles)
        lidar_y = np.arrya(objectRange)*np.sin(angles)
        lambda_ = 10*(np.pi/180)
        sigma_r = 0.005

        breaking_index = []
        delta_pi = lidar_data.angle_increment
        for i in range(size_of_range):
            if i == 0:
                continue 
            threshold_ = (objectRange[i-1]*np.sin(delta_pi)/ np.sin(lambda_-delta_pi))+ 3*sigma_r
            if abs(objectRange[i]-objectRange[i-1] > threshold_):
                breaking_index.append(i)

        U_k = 1.0
        K_f = np.zeros([size_of_range,1])
        K_b = np.zeros([size_of_range,1])
        theta_ = np.zeros([size_of_range,1])
        
        for i in range(size_of_range):
            for K_f_tmp in range(1,size_of_range-i):
                Eucli_dis = np.sqrt(pow(lidar_x[i]-lidar_x[K_f_tmp+i],2)+pow(lidar_y[i]-lidar_y[K_f_tmp+i],2))
                real_dis = 0
                for j in range(K_f_tmp):
                    real_dis = real_dis + np.sqrt(pow(lidar_x[i]-lidar_x[j+i],2)+pow(lidar_y[i]-lidar_y[j+i],2))

                if Eucli_dis > real_dis-U_k:
                    K_f[i] = K_f_tmp
                    f_i = np.array([lidar_x[i+K_f[i]]-lidar_x[i], lidar_y[i+K_f[i]]-lidar_y[i]])
                else:
                    break
            
            for K_b_tmp in range(1,i+1):
                Eucli_dis = np.sqrt(pow(lidar_x[i]-lidar_x[-K_b_tmp+i],2)+pow(lidar_y[i]-lidar_y[-K_b_tmp+i],2))
                real_dis = 0
                for j in range(K_b_tmp):
                    real_dis = real_dis + np.sqrt(pow(lidar_x[i]-lidar_x[-j+i],2)+pow(lidar_y[i]-lidar_y[-j+i],2))

                if Eucli_dis > real_dis-U_k:
                    K_b[i] = K_b_tmp
                    b_i = np.array([lidar_x[i-K_b[i]]-lidar_x[i], lidar_y[i-K_b[i]]-lidar_y[i]])
                else:
                    break
            
            theta_[i] = np.arccos(np.dot(f_i,b_i)/(np.linalg.norm(f_i)*np.linalg.norm(b_i)))

        corner_index = theta_.index(min(theta_))

        if len(breaking_index) == 0:
            l1 = np.array([lidar_x[0]-lidar_x[corner_index], lidar_y[0]-lidar_y[corner_index]])
            l2 = np.array([-lidar_x[corner_index],-lidar_y[corner_index]])
            theta = np.arccos(np.dot(l1,l2)/(np.linalg.norm(l1)*np.linalg.norm(l2)))
            current_x = np.linalg.norm(l2)*np.cos(theta)
            current_y = np.linalg.norm(l2)*np.sin(theta)
        
        elif len(breaking_index) == 1:
            if breaking_index[0] < corner_index:
                l1 = np.array([lidar_x[breaking_index[0]+1]-lidar_x[corner_index], lidar_y[breaking_index[0]+1]-lidar_y[corner_index]])
                l2 = np.array([-lidar_x[corner_index],-lidar_y[corner_index]])
                theta = np.arccos(np.dot(l1,l2)/(np.linalg.norm(l1)*np.linalg.norm(l2)))
                current_x = np.linalg.norm(l2)*np.cos(theta)
                current_y = np.linalg.norm(l2)*np.sin(theta)
            else:
                l1 = np.array([lidar_x[0]-lidar_x[corner_index], lidar_y[0]-lidar_y[corner_index]])
                l2 = np.array([-lidar_x[corner_index],-lidar_y[corner_index]])
                theta = np.arccos(np.dot(l1,l2)/(np.linalg.norm(l1)*np.linalg.norm(l2)))
                current_x = np.linalg.norm(l2)*np.cos(theta)
                current_y = np.linalg.norm(l2)*np.sin(theta)
        else:
            target_index = 0
            for index in breaking_index:
                if index < corner_index and index > target_index:
                    target_index = index
                elif index > corner_index:
                    break
            l1 = np.array([lidar_x[target_index]-lidar_x[corner_index], lidar_y[target_index]-lidar_y[corner_index]])
            l2 = np.array([-lidar_x[corner_index],-lidar_y[corner_index]])
            theta = np.arccos(np.dot(l1,l2)/(np.linalg.norm(l1)*np.linalg.norm(l2)))
            current_x = np.linalg.norm(l2)*np.cos(theta)
            current_y = np.linalg.norm(l2)*np.sin(theta)

        rospy.loginfo("current_x : " + str(current_x) + " current_y : " + str(current_y))
        return [current_x, current_y]
    

    

        

    def _write_message(self, detection_results, boxes, scores, classes):
        """ populate output message with input header and bounding boxes information """
        if boxes is None:
            return None
        # if self.print_outcome:
            # rospy.loginfo("pc_list len = " + str(len(boxes)))

        self.class_dict_tmp = {}

        for box, score, category in zip(boxes, scores, classes):
            # Populate darknet message
            left, bottom, right, top = box
            detection_msg = BoundingBox()
            detection_msg.xmin = left
            detection_msg.xmax = right
            detection_msg.ymin = top
            detection_msg.ymax = bottom
            detection_msg.probability = score
            detection_msg.Class = category
            detection_results.bounding_boxes.append(detection_msg)

            if category == "landing":
                # self.extractingObjectFromLidarLanding(self.lidar_data,left,right)
                return detection_results
            y_center = int(bottom - ((bottom - top) / 2))
            x_center = int(right - ((right - left) / 2))       
          
            if self.lidar_sub_switch is True:  

                if 'dog' in category: 
                    pc_list = self.extractingObjectFromLidarDog(self.lidar_data,left,right)
                else:            
                    pc_list = self.extractingObjectFromLidar(self.lidar_data,left,right)
    
                if len(pc_list) > 0:                    
                    obj_pose_x, obj_pose_y= pc_list
                    obj_pose_z = 0
                    rospy.loginfo("x  = " + str(obj_pose_x)+ "  y  = " + str(obj_pose_y)+"  z  = " + str(obj_pose_z))                    
                    object_tf =  [obj_pose_x, obj_pose_y, obj_pose_z] 
                    tf_id = str(category)                    
                    self._tfpub.sendTransform((object_tf),
                                                    tf.transformations.quaternion_from_euler(
                                                        0, 0, 0),
                                                    rospy.Time.now(),
                                                    tf_id,
                                                    'camera_link')
                                                    
                    self.object_position = self.transMat.dot(np.array([obj_pose_x, obj_pose_y, obj_pose_z, 1]))
                    self.object_pose = PoseStamped()
                    self.object_pose.header.seq = self.corresponding_seq
                    self.object_pose.header.stamp = rospy.Time.now()
                    self.object_pose.header.frame_id = tf_id
                    self.object_pose.pose.position.x = self.object_position[0]
                    self.object_pose.pose.position.y = self.object_position[1]
                    self.object_pose.pose.position.z = self.object_position[2]
                    self.object_pose.pose.orientation.x = 0.0
                    self.object_pose.pose.orientation.y = 0.0
                    self.object_pose.pose.orientation.z = 0.0
                    self.object_pose.pose.orientation.w = 1.0

                    self._pub_object.publish(self.object_pose)
                    quaternion = (self.drone_pose_map.pose.orientation.x,self.drone_pose_map.pose.orientation.y,self.drone_pose_map.pose.orientation.z,self.drone_pose_map.pose.orientation.w)
                    euler = tf.transformations.euler_from_quaternion(quaternion)
                    roll = euler[0]
                    pitch = euler[1]
                    yaw = euler[2] + np.pi/4
                    self.object_y_map = 2.0 + self.drone_pose_map.pose.position.x + (self.object_pose.pose.position.x*np.cos(yaw))-(self.object_pose.pose.position.y*np.sin(yaw))
                    self.object_x_map = 2.0 + self.drone_pose_map.pose.position.y + (self.object_pose.pose.position.x*np.sin(yaw))+(self.object_pose.pose.position.y*np.cos(yaw))
                    
                    if tf_id in self.class_dict_tmp.keys():
                        self.class_dict_tmp[tf_id].append([self.object_x_map, self.object_y_map])
                    else:
                        self.class_dict_tmp[tf_id] = [[self.object_x_map, self.object_y_map]]
                    
        return detection_results

    @timeit_ros
    def process_frame(self):
        """ Main function to process the frame and run the infererence """
        # Deque the next image msg
        current_msg = self.msg_queue.get()
        current_image = None
        # Convert to image to OpenCV format
        try:
            current_image = self._bridge.imgmsg_to_cv2(current_msg, "bgr8")
            rospy.logdebug("[trt_yolo_ros] image converted for processing")
        except CvBridgeError as e:
            rospy.logdebug("Failed to convert image %s", str(e))
        # Initialize detection results
        if current_image is not None:
            rospy.logdebug("[trt_yolo_ros] processing frame")
            boxes, classes, scores, visualization = self.model(current_image)
            detection_results = BoundingBoxes()
            detection_results.header = current_msg.header
            detection_results.image_header = current_msg.header
            # construct message
            self._write_message(detection_results, boxes, scores, classes)

            # Save the Image
            if boxes is not None and not self.landingSign.data:
                for label, score in zip(classes, scores):
                    if label == "landing":
                        rospy.loginfo("Landing Detected  // Score : " + str(score))
                        if score > 0.7 and self.RTBSign: 
                            rospy.loginfo("Send landing Sign")
                            self.landingSign.data = True
                            self._pub_landing_sign.publish(self.landingSign)

                        continue
                    if label in self.class_dict.keys():
                        for index_tmp, pose_tmp in enumerate(self.class_dict_tmp[label]): 
                            is_matching = False 
                            for index, pose in enumerate(self.class_dict[label]):
                                if (pow((pose[0]-pose_tmp[0]),2) + pow((pose[1]-pose_tmp[1]),2)) < 4:
                                    rospy.loginfo("Looks like same -> distance : " + str(pow((pose[0]-pose_tmp[0]),2) + pow((pose[1]-pose_tmp[1]),2))+" between "+str(index) +" and "+str(index_tmp))
                                    self.class_dict[label][index] = self.class_dict_tmp[label][index_tmp]
                                    # save the image that has have the best score === NEED TO ADD IT
                                    cv2.imwrite(self.IMAGE_PATH+label+str(index)+".jpg",visualization)
                                    is_matching = True
                                    f = open(self.TEXT_PATH+label+str(index)+".txt",'a')
                                    f.write(str(pose_tmp[0])+" "+str(pose_tmp[1])+"\n")
                                    f.close()

                            if not is_matching:
                                #this pose seems to be the new one : same class but different position
                                self.class_dict[label].append(self.class_dict_tmp[label][index_tmp])
                                f = open(self.TEXT_PATH+label+str(len(self.class_dict[label])-1)+".txt",'w')
                                f.write(str(pose_tmp[0]) +" "+ str(pose_tmp[1])+"\n")
                                f.close()
                    else:
                        self.class_dict[label] = []
                        self.class_dict[label].append(self.class_dict_tmp[label][0])
                        cv2.imwrite(self.IMAGE_PATH+label+str(0)+".jpg",visualization)
                        f = open(self.TEXT_PATH+label+str(0)+".txt",'w')
                        f.write(str(self.class_dict_tmp[label][0][0]) +" "+ str(self.class_dict_tmp[label][0][1])+"\n")
                        f.close()
            try:
                rospy.logdebug("[trt_yolo_ros] publishing")
                self._pub.publish(detection_results)
                
                if self.publish_image:
                    self._pub_viz.publish(
                        self._bridge.cv2_to_imgmsg(visualization, "bgr8")
                    )
            except CvBridgeError as e:
                rospy.logdebug("[trt_yolo_ros] Failed to convert image %s", str(e))
