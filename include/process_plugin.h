#pragma once

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/callback_queue.h>
#include <thread>
#include <dynamic_reconfigure/server.h>
#include <rm_msgs/TargetDetectionArray.h>
#include <Eigen/Dense>
#include <rm_msgs/TrackData.h>
#include <boost/bind.hpp>
#include <opencv2/core/ocl.hpp>

#include "inference.h"
#include "common.h"
#include "rm_digitalimg_proc_deep/InferenceConfig.h"

namespace rm_digitalimg_proc_deep {

class Processor : public nodelet::Nodelet, public deepProcess
{
public:
    Processor() = default;
    ~Processor() override
    {
        if (this->my_thread_.joinable())

            this->my_thread_.join();
    }

    void onInit() override;
    void initialize(ros::NodeHandle &nh);
    // callback
    void detectionCB(const rm_msgs::TargetDetectionArray::ConstPtr &msg);
    void trackCB(const rm_msgs::TrackData &track_data);
    void computeCB(const rm_msgs::TrackData &track_data);
    void inferenceconfigCB(InferenceConfig &config, uint32_t level);
    void draw();
    void callback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info);

private:
    ros::NodeHandle nh_;
    std::shared_ptr<image_transport::ImageTransport> it_;
    std::thread my_thread_;

    int target_is_red_{};

    // detection, compute, track
    rm_msgs::TargetDetection detection_;
    cv::Mat_<double> r_vec_ = cv::Mat_<double>(3, 1);
    cv::Mat_<double> t_vec_ = cv::Mat_<double>(3, 1);

    // track
    rm_msgs::TargetDetectionArray target_array_;
    std::vector<geometry_msgs::PointStamped> all_points_;

    // compute
    geometry_msgs::PointStamped compute_point_;

    // dynamic reconfigure
    std::unique_ptr<dynamic_reconfigure::Server<InferenceConfig>> inference_cfg_srv_;
    dynamic_reconfigure::Server<InferenceConfig>::CallbackType inference_cfg_cb_;

    // dynamic parameter
    float confidence_threshold_;
    float nms_threshold_;
    TargetColor target_color_;
    DrawType draw_type_;

    // pub
    image_transport::Publisher img_pub_;
    ros::Publisher target_pub_;

    // sub
    image_transport::CameraSubscriber cam_sub_;
    ros::Subscriber track_sub_;
    ros::Subscriber detection_sub_;
    ros::Subscriber compute_sub_;

    // tf
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // camera
    sensor_msgs::CameraInfoConstPtr camera_info_{};
};

}

// 这里把is_red和is_large给ban掉了，后续出错再改0
