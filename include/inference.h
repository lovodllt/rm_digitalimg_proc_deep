#pragma once
#include <common.h>
#include <mutex>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <image_transport/image_transport.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <openvino/runtime/runtime.hpp>

#define model_path "/home/lovod/rm_code/src/rm_visplugin/rm_digitalimg_proc_deep/model/v12_openvino_model/v12.xml"
#define target_size 640

static const std::vector<std::string> colors = {"blue", "red"};

// 图像处理
struct dataImg {
    cv::Mat input;     // 原始输入图像
    float scale;       // 缩放比例
    int pad_left;      // 填充宽度
    int pad_top;       // 填充高度
    cv::Mat blob;      // 处理后的图像blob
};
// 初步筛选后装甲板
struct preArmor {
    std::vector<float> cls_conf;
    std::vector<cv::Rect> boxes;
};
// 灯条
struct Bar {
    cv::RotatedRect rect;
    cv::Point2f center;
    float ratio;
    float angle;
    std::vector<cv::Point2f> sorted_points;
    double long_one;
};
// 模型推理后装甲板
struct inferredArmor {
    float cls_conf;
    cv::Rect2f box;
    int color;// 1:blue 2:red
    cv::Mat num_roi;
    cv::Mat hsv_roi;
    cv::Point2f roi_position;
    std::vector<Bar> bars;
};
// 最终装甲板
struct finalArmor {
    float cls_conf;
    std::vector<cv::Point2f> armor_points;
    cv::Point2f center;
    std::string label;
    int color;// 1:blue 2:red
    cv::Mat num_roi;
};

class deepProcess {
public:
    deepProcess()
    {
        using_once();
    }

    void using_once();
    void init();

    dataImg preprocess_img(cv::Mat &raw_img);
    std::vector<finalArmor> InferAndPostprocess(dataImg &img_data);
    void colorFiliter(cv::Mat &img, inferredArmor &armor, std::vector<inferredArmor> &qualifiedArmors, TargetColor target_color);
    bool isValidBar(Bar &bar);
    bool isValidArmor(inferredArmor &armor);
    void barFiliter(std::vector<inferredArmor> &qualifiedArmors);
    void armorFiliter(std::vector<inferredArmor> &qualifiedArmors, std::vector<finalArmor> &final_armors);

    void show_bar(cv::Mat &img, std::vector<inferredArmor> &qualifiedArmors);
    void show_box(cv::Mat &img, std::vector<finalArmor> &finalArmors);
    void show_track(cv::Mat &img, sensor_msgs::CameraInfoConstPtr &camera_info_, std::vector<geometry_msgs::PointStamped> all_points_, geometry_msgs::PointStamped compute_point_);

private:
    ov::Core core;
    std::once_flag flag;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Output<const ov::Node> input_port;
    ov::Tensor input_tensor;

protected:
    // target
    int target_is_red_{};
    bool is_large_armor_ = false;

    // inference
    std::vector<inferredArmor> qualifiedArmors;
    std::vector<finalArmor> finalArmors;

    // dynamic parameter
    // inference
    double confidence_threshold_{};
    double nms_threshold_{};

    // preprocess
    double gamma_{};
    double l_mean_threshold_{};

    TargetColor target_color_{};
    DrawType draw_type_{};
};

// HSV
#define red_h_max_low_       25
#define red_h_min_low_       0
#define red_h_max_high_      180
#define red_h_min_high_      140
#define red_s_max_           255
#define red_s_min_           40
#define red_v_max_           255
#define red_v_min_           140
#define blue_h_max_          130
#define blue_h_min_          80
#define blue_s_max_          255
#define blue_s_min_          30
#define blue_v_max_          255
#define blue_v_min_          190

// BAR
#define min_lw_ratio_        2
#define max_lw_ratio_        10
#define min_bar_angle_       55

// ARMOR
#define max_bars_ratio_      1.5
#define min_bars_distance_   0.4
#define max_bars_distance_   2
#define max_bars_angle_      45