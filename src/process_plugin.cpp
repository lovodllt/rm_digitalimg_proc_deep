#include "process_plugin.h"

namespace rm_digitalimg_proc_deep {

void Processor::onInit()
{
    ros::NodeHandle &nh = getMTPrivateNodeHandle();
    static ros::CallbackQueue my_queue;
    nh.setCallbackQueue(&my_queue);

    initialize(nh);

    my_thread_ = std::thread([this](){
        ros::SingleThreadedSpinner spinner;
        spinner.spin(&my_queue);
    });
}

void Processor::initialize(ros::NodeHandle &nh)
{
    nh_ = ros::NodeHandle(nh, "digitalimg_proc_deep");
    auto inference_params_init = [this, &nh]()
    {
        ROS_INFO("reading inference param");
        confidence_threshold_ = nh.param("score_threshold", decltype(confidence_threshold_){0.3});
        nms_threshold_ = nh.param("nms_threshold", decltype(nms_threshold_){0.3});
        int target_color = nh.param("target_color", static_cast<int>(TargetColor::BLUE));
        target_color_ = static_cast<TargetColor>(target_color);
        int draw_type = nh.param("draw_type", static_cast<int>(DrawType::RAW));
        draw_type_ = static_cast<DrawType>(draw_type);
        ROS_INFO("inference params reading done");

        ROS_INFO("confidence_threshold_: %f", confidence_threshold_);
        ROS_INFO("nms_threshold_: %f", nms_threshold_);
        ROS_INFO("target_color_: %d", target_color_);
        ROS_INFO("draw_type_: %d", draw_type_);
    };

    //inference_params_init();

    // 创建动态配置服务器并设置回调函数
    inference_cfg_srv_ = std::make_unique<dynamic_reconfigure::Server<InferenceConfig>>(ros::NodeHandle(nh_, "inference_condition"));
    inference_cfg_cb_ = boost::bind(&Processor::inferenceconfigCB, this, _1, _2);
    inference_cfg_srv_->setCallback(inference_cfg_cb_);

    // tf相关初始化
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(10));
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // it_用于创建图像传输对象，用于发布和订阅图像消息(相比于advertise<sensor_msgs::Image>，在优化速率的同时可以处理多种图像压缩格式)
    it_ = std::make_shared<image_transport::ImageTransport>(nh_);
    img_pub_ = it_->advertise("debug_image", 1);
    cam_sub_ = it_->subscribeCamera("/galaxy_camera/image_raw", 10, &Processor::callback, this);
    target_pub_ = nh.advertise<decltype(target_array_)>("/processor/result_msg", 10);
}

// 动态配置服务器的回调函数
void Processor::inferenceconfigCB(InferenceConfig &config, uint32_t level)
{
    confidence_threshold_ = config.confidence_threshold;
    nms_threshold_ = config.nms_threshold;

    gamma_ = config.gamma;
    l_mean_threshold_ = config.l_mean_threshold;

    target_color_ = static_cast<TargetColor>(config.target_color);
    draw_type_ = static_cast<DrawType>(config.draw_type);

    if (draw_type_ != DrawType::TRACK)
    {
        detection_sub_.shutdown();
        track_sub_.shutdown();
        compute_sub_.shutdown();
    }
    else
    {
        detection_sub_ = nh_.subscribe<rm_msgs::TargetDetectionArray>("/detection", 10, &Processor::detectionCB, this);
        track_sub_ = nh_.subscribe("/track", 1, &Processor::trackCB, this);
        compute_sub_ = nh_.subscribe("/compute_target_position", 1, &Processor::computeCB, this);
    }
}

// "/detection"的回调函数，获取目标姿态信息，转换为rvec和tvec
void Processor::detectionCB(const rm_msgs::TargetDetectionArray::ConstPtr &msg)
{
    if (!msg->detections.empty())
    {
        detection_ = msg->detections.front();
        // 获取四元数
        Eigen::Quaterniond q(
            detection_.pose.orientation.w,
            detection_.pose.orientation.x,
            detection_.pose.orientation.y,
            detection_.pose.orientation.z);

        // 转换为旋转向量
        constexpr double MIN_NORM = 1e-6;
        if (q.norm() > MIN_NORM)
        {
            Eigen::AngleAxisd rotation_vec(q);
            Eigen::Vector3d eigen_rvec = rotation_vec.angle() * rotation_vec.axis();
            // 转换为OpenCV格式的旋转向量
            r_vec_ = cv::Mat(3, 1, CV_64F, eigen_rvec.data());
        }
        else
        {
            r_vec_.setTo(0);
        }

        // 转换为平移向量
        t_vec_ << detection_.pose.position.x,
                  detection_.pose.position.y,
                  detection_.pose.position.z;
    }
}

// "/track"的回调函数，处理目标跟踪信息
void Processor::trackCB(const rm_msgs::TrackData &track_data)
{
    all_points_.clear();

    // 记录数据
    double yaw = track_data.yaw, r1 = track_data.radius_1, r2 = track_data.radius_2;
    double xc = track_data.position.x, yc = track_data.position.y, zc = track_data.position.z;
    double dz = track_data.dz;  // 相邻装甲板的高度差
    int armor_num = track_data.armors_num;
    geometry_msgs::PointStamped armor_position;
    geometry_msgs::PointStamped car_position;
    double r = 0;
    bool is_current_pair = true;

    // 获取坐标转换信息
    geometry_msgs::TransformStamped transformStamped = tf_buffer_->lookupTransform(
        "camera_optical_frame", track_data.header.frame_id, track_data.header.stamp, ros::Duration(1));

    // 装甲板的坐标变换
    for (int i = 0; i < 4; i++)
    {
        // 每个装甲板的间的角度间隔
        double tmp_yaw = yaw + i * (2 * M_PI / armor_num);
        if (armor_num == 4)
        {
            r = is_current_pair ? r1 : r2;
            armor_position.point.z = zc + (is_current_pair ? 0 : dz);
            is_current_pair = !is_current_pair;
        }
        else
        {
            r = r1;
            armor_position.point.z = zc;
        }
        armor_position.point.x = xc - r * cos(tmp_yaw);
        armor_position.point.y = yc - r * sin(tmp_yaw);
        armor_position.header = track_data.header;

        try
        {
            // 应用变换到点上
            tf2::doTransform(armor_position, armor_position, transformStamped);
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("Error: %s", e.what());
            return;
        }

        all_points_.push_back(armor_position);
    }

    // 车体的坐标变换
    // car_position.point.x = xc;
    // car_position.point.y = yc;
    // car_position.point.z = zc;
    //
    // try
    // {
    //     // 应用变换到点上
    //     tf2::doTransform(car_position, car_position, transformStamped);
    // }
    // catch (const std::exception &e)
    // {
    //     ROS_ERROR("Error: %s", e.what());
    //     return;
    // }
    //
    // all_points_.push_back(car_position);

    armor_position.point.x = xc;
    armor_position.point.y = yc;
    armor_position.point.z = zc;

    try
    {
        // 应用变换到点上
        tf2::doTransform(armor_position, armor_position, transformStamped);
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("Error: %s", e.what());
        return;
    }

    all_points_.push_back(armor_position);
}

// "/compute_target_position"的回调函数，
void Processor::computeCB(const rm_msgs::TrackData &track_data)
{
    compute_point_.point = track_data.position;
    try
    {
        geometry_msgs::TransformStamped transformStamped = tf_buffer_->lookupTransform(
            "camera_optical_frame", track_data.header.frame_id, track_data.header.stamp, ros::Duration(1));
        // 应用变换到点上
        tf2::doTransform(compute_point_, compute_point_, transformStamped);
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("Error: %s", e.what());
        return;
    }
}

// 图像绘制及发布
void Processor::draw(cv::Mat &img)
{
    if (draw_type_ == DrawType::RAW) {}
    else if (draw_type_ == DrawType::ARMOR)
    {
        show_box(img, finalArmors);
    }
    else if (draw_type_ == DrawType::TRACK)
    {
        show_box(img, finalArmors);
        show_track(img, camera_info_, all_points_, compute_point_);
    }

    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    img_pub_.publish(img_msg);
}

// 相机回调函数(主函数)
void Processor::callback(const sensor_msgs::ImageConstPtr &img, const sensor_msgs::CameraInfoConstPtr &info)
{
    this->target_array_.detections.clear();
    this->qualifiedArmors.clear();
    this->finalArmors.clear();

    try
    {
        // 记录相机信息
        camera_info_ = info;
        target_array_.header = info->header;

        // 将ROS图像消息转换为OpenCV图像
        cv::Mat frame = cv_bridge::toCvShare(img, "bgr8")->image;

        // 求图像中心
        float img_w = frame.cols;
        float img_h = frame.rows;
        float img_center_x = img_w / 2;
        float img_center_y = img_h / 2;

        // 图像预处理
        dataImg img_data = preprocess_img(frame);

        // 装甲板识别
        finalArmors = InferAndPostprocess(img_data);
        classify(finalArmors);

        // 将finalarmor信息转换为track
        for (auto &armor : finalArmors)
        {
            cv::Point2f center = armor.center;
            double distanceToImg = sqrt(pow(center.x - img_center_x, 2) + pow(center.y - img_center_y, 2));

            rm_msgs::TargetDetection target;

            // 设置置信度和装甲板中心到图像中心的距离
            target.confidence = armor.cls_conf;
            target.distance_to_image_center = distanceToImg;

            // 序列化装甲板角点
            int32_t temp[8];
            for (int i = 0; i < 4; i++)
            {
                temp[i * 2] = static_cast<int>(armor.armor_points[i].x);
                temp[i * 2 + 1] = static_cast<int>(armor.armor_points[i].y);
            }

            // 将角点信息复制到one_target的pose.orientation中(利用四元数的4个float32存储8个int32)
            memcpy(&target.pose.orientation.x, &temp[0], sizeof(int32_t) * 2);
            memcpy(&target.pose.orientation.y, &temp[2], sizeof(int32_t) * 2);
            memcpy(&target.pose.orientation.z, &temp[4], sizeof(int32_t) * 2);
            memcpy(&target.pose.orientation.w, &temp[6], sizeof(int32_t) * 2);

            target_array_.detections.push_back(target);
        }

        for (auto &target : target_array_.detections)
        {
            target.pose.position.x = info->roi.x_offset;
            target.pose.position.y = info->roi.y_offset;
        }
        target_array_.is_red = target_is_red_;

        target_pub_.publish(target_array_);

        // 绘制及发布图像
        if (draw_type_ != DrawType::DISABLE)
        {
            draw(frame);
        }
    }
    catch (const cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

}

PLUGINLIB_EXPORT_CLASS(rm_digitalimg_proc_deep::Processor, nodelet::Nodelet)
