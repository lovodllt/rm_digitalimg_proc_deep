#include "number_classifier.h"

void number_classifier::init()
{
    std::cout<<"initialize"<<std::endl;

    net_ = cv::dnn::readNetFromONNX(number_classifier_model_path_);
}

// 确保初始化操作仅被执行一次
void number_classifier::using_once()
{
    std::call_once(flag_, [this](){this->init();});
}

void number_classifier::warp(finalArmor &armor)
{
    const int top_light_y_ = (warp_height_ - light_len_) / 2;
    const int bottom_light_y_ = top_light_y_ + light_len_;

    cv::Point2f src_pts[4] = {armor.armor_points[0], armor.armor_points[1], armor.armor_points[2], armor.armor_points[3]};

    cv::Point2f dst_pts[4] = {
        cv::Point2f(0, bottom_light_y_),
        cv::Point2f(0, top_light_y_),
        cv::Point2f(warp_width_, top_light_y_),
        cv::Point2f(warp_width_, bottom_light_y_),
    };

    std::cout << "Src Points: ";
    for (auto& pt : src_pts) std::cout << pt << " ";
    std::cout << "\nDst Points: ";
    for (auto& pt : dst_pts) std::cout << pt << " ";
    std::cout << std::endl;

    auto warp_mat = getPerspectiveTransform(src_pts, dst_pts);
    warp_mat.at<double>(0, 2) /= 1000.0;
    warp_mat.at<double>(1, 2) /= 1000.0;
    std::cout << "Warp Matrix: " << warp_mat << std::endl;

    try
    {
        warpPerspective(armor.num_roi, armor.num_roi, warp_mat, cv::Size(warp_width_, warp_height_), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    catch (const cv::Exception &e)
    {
        ROS_ERROR("warp err occurred !");
        ROS_ERROR("cv_exception: %s", e.what());
    }
}

// 提取数字图像
cv::Mat number_classifier::extractNumbers(finalArmor &armor)
{
    //warp(armor);
    cv::Mat num_img;

    int roi_w = armor.num_roi.cols;
    int roi_h = armor.num_roi.rows;
    int crop_x = static_cast<int>(roi_w * 0.2);
    int crop_y = 0;
    int crop_w = static_cast<int>(roi_w * 0.5);
    int crop_h = roi_h;

    crop_x = std::max(0, crop_x);
    crop_w = std::min(crop_w, roi_w - crop_x);

    cv::Rect roi(crop_x, crop_y, crop_w, crop_h);
    cv::Mat crop_img = armor.num_roi(roi);

    resize(crop_img, num_img, cv::Size(roi_w_, roi_h_));

    // 二值化处理
    threshold(num_img, num_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    return num_img;
}

// 数字分类
void number_classifier::classify(std::vector<finalArmor> &finalArmors)
{
    // 使用迭代器循环，避免范围for修改容器导致的迭代器失效
    for (auto it = finalArmors.begin(); it != finalArmors.end();)
    {
        auto &armor = *it;  // 通过迭代器访问当前元素
        cv::Mat num_mat, num_blob;

        num_mat = extractNumbers(armor);

        if (num_mat.empty())
        {
            it = finalArmors.erase(it);
            continue;
        }

        imshow("number", num_mat);
        cv::dnn::blobFromImage(num_mat, num_blob);

        net_.setInput(num_blob);
        cv::Mat outputs = net_.forward();

        // softmax处理
        float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
        cv::Mat softmax_prob;
        cv::exp(outputs - max_prob, softmax_prob);
        float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
        softmax_prob /= sum;

        // 获取分类结果
        double confidence;
        cv::Point class_id_point;
        minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
        int label_id = class_id_point.x;

        // 更新装甲板信息(如果装甲板不包含label信息，则删除)
        std::string label = class_names_[label_id];
        if (label != "negative")
        {
            armor.label = label;
            ++it;
        }
        else
        {
            it = finalArmors.erase(it);
            std::cout<<"number wrong"<<std::endl;
        }
    }
}