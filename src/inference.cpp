#include "inference.h"

#include <common.h>

#include "number_classifier.h"
static number_classifier num_classifier;

void deepProcess::init()
{
    std::cout<<"initialize"<<std::endl;

    // // 配置GPU插件参数
    // ov::AnyMap config = {
    //     {ov::hint::inference_precision(ov::element::f16)},                   // 指定FP16精度
    //     {ov::hint::allow_auto_batching(true)},                               // 允许动态形状
    //     {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}, // 强制使用显存
    //     {ov::hint::num_requests(4)}                                          // 设置推理核心数
    // };

    // 加载模型
    compiled_model = core.compile_model(model_path, "GPU");
    // 创建推理请求
    infer_request = compiled_model.create_infer_request();
    // 创建输入端口
    input_port = compiled_model.input();
}

// 确保初始化操作仅被执行一次
void deepProcess::using_once()
{
    std::call_once(flag, [this](){this->init();});
}

// 图像预处理
dataImg deepProcess::preprocess_img(const cv::UMat& img)
{
    // 0.自动提亮
    cv::UMat hsv;
    cvtColor(img, hsv, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    split(hsv, channels);

    double l_mean = mean(channels[0])[0];
    std::cout<<"l_mean: "<<l_mean<<std::endl;

    if (l_mean < 10)
    {
        double gamma = 0.7;
        cv::Mat lut(1, 256, CV_8UC1);
        for (int i = 0; i < 256; i++)
        {
            lut.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        LUT(channels[0], lut, channels[0]);
    }

    cv::UMat bgr_img;
    cv::merge(channels, hsv);
    cvtColor(hsv, bgr_img, cv::COLOR_Lab2BGR);

    hsv.release();
    channels.clear();

    // 1.获取图像尺寸
    int h = bgr_img.rows;
    int w = bgr_img.cols;
    int th = target_size;
    int tw = target_size;

    // 2.计算缩放比例
    float scale = std::min(static_cast<float>(tw) / w, static_cast<float>(th) / h);
    scale = std::max(scale, 0.01f);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    // 3.缩放图像
    cv::UMat resized_img;
    resize(bgr_img, resized_img, cv::Size(new_w, new_h));

    bgr_img.release();

    // 4.计算填充量
    int padW = tw - new_w;
    int padH = th - new_h;

    int left = padW / 2;
    int right = padW - left;
    int top = padH / 2;
    int bottom = padH - top;

    // 5.填充图像
    cv::UMat padded_img;
    copyMakeBorder(resized_img, padded_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // 6.创建blob
    cv::Mat blob = cv::dnn::blobFromImage(padded_img, 1.0/255.0, cv::Size(target_size, target_size), cv::Scalar(0,0,0), true, CV_16F);

    // 7.返回结果
    dataImg data;
    data.input = img;
    data.scale = scale;
    data.pad_left = left;
    data.pad_top = top;
    data.blob = blob;

    bgr_img.release();
    resized_img.release();
    padded_img.release();

    return data;
}

// 模型推理(主函数)
std::vector<finalArmor> deepProcess::InferAndPostprocess(dataImg &imgdata, float score_threshold_, float nms_threshold_, TargetColor target_color)
{
    // 1.初始化模型
    using_once();

    // 2.创建输入张量 (获取输入图像格式，维度格式[N,C,H,W]，预处理后的图像指针)
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), imgdata.blob.ptr());
    // 3.设置输入张量
    infer_request.set_input_tensor(input_tensor);

    // 4.执行推理并获取推理结果
    infer_request.infer();
    auto output = infer_request.get_output_tensor(0); // ouput[cx,cy,w,h,cls_conf]
    auto output_shape = output.get_shape();

    // 5.重新设定输出张量形状(对于最终的output_buffer[8400,5],即每行为一个检测框，对应5列数据)
    auto data = output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer);

    // 6.遍历每个检测框
    preArmor preArmors;
    for (int i = 0; i < output_buffer.rows; i++)
    {
        float cls_conf = output_buffer.at<float>(i, 4);
        // 过滤掉置信度小于阈值的检测框
        if (cls_conf > score_threshold_)
        {
            // 提取边界框信息
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);

            // 将坐标映射回原始图像
            float cx_unpad = (cx - imgdata.pad_left) / imgdata.scale;
            float cy_unpad = (cy - imgdata.pad_top) / imgdata.scale;
            float w_unpad = w / imgdata.scale;
            float h_unpad = h / imgdata.scale;

            // 计算边界框的左上角和宽高(压缩roi高度避免干扰)
            int lx = static_cast<int>(cx_unpad - w_unpad / 2 - 5);
            int ly = static_cast<int>((cy_unpad - h_unpad / 2));
            int width = static_cast<int>(w_unpad + 10);
            int height = static_cast<int>(h_unpad);

            // 边界检查
            if (lx < 0 || ly < 0  || lx + width > imgdata.input.cols || ly + height > imgdata.input.rows)
            {
                continue;
            }

            cv::Rect box(lx, ly, width, height);

            preArmors.boxes.push_back(box);
            preArmors.cls_conf.push_back(cls_conf);
        }
    }

    // 7.NMS处理
    std::vector<int> indices;
    cv::dnn::NMSBoxes(preArmors.boxes, preArmors.cls_conf, score_threshold_, nms_threshold_, indices);

    // 8.传统处理角点
    for (auto i : indices)
    {
        inferredArmor infer_armor;

        infer_armor.box = preArmors.boxes[i];
        infer_armor.cls_conf = preArmors.cls_conf[i];

        std::cout<<"model find armors"<<std::endl;
        std::cout<<"box: "<<infer_armor.box<<"    cls_conf: "<<infer_armor.cls_conf<<std::endl;

        colorFiliter(imgdata.input, infer_armor, qualifiedArmors, target_color);
    }
    barFiliter(qualifiedArmors);
    armorFiliter(qualifiedArmors, finalArmors);
    num_classifier.classify(finalArmors);

    return finalArmors;
}

// 判断装甲板颜色
void deepProcess::colorFiliter(cv::UMat& img, inferredArmor &right_armor, std::vector<inferredArmor> &qualifiedArmors, TargetColor target_color_)
{
    std::cout<<"target_color: "<<target_color_<<std::endl;
    cv::UMat roi = img(right_armor.box);

    // 存储单通道roi
    cv::UMat num_img;
    cvtColor(roi, num_img, cv::COLOR_BGR2GRAY);
    right_armor.num_roi = num_img.clone();

    // 转换hsv图像
    cv::UMat hsv_img;
    cvtColor(roi, hsv_img, cv::COLOR_BGR2HSV);

    cv::UMat binary_image_;
    if(target_color_ == 1)
    {
        inRange(hsv_img, cv::Scalar(blue_h_min_, blue_s_min_, blue_v_min_), cv::Scalar(blue_h_max_, blue_s_max_, blue_v_max_), binary_image_);
        int nonZeroCount = countNonZero(binary_image_);
        if(nonZeroCount > 30)
        {
            right_armor.color = 1;
            right_armor.hsv_roi = binary_image_.clone();
            qualifiedArmors.push_back(right_armor);
        }
    }
    else if(target_color_ == 2)
    {
        cv::UMat h_binary_low, h_binary_high;
        inRange(hsv_img, cv::Scalar(red_h_min_low_, red_s_min_, red_v_min_), cv::Scalar(red_h_max_low_, red_s_max_, red_v_max_), h_binary_low);
        inRange(hsv_img, cv::Scalar(red_h_min_high_, red_s_min_, red_v_min_), cv::Scalar(red_h_max_high_, red_s_max_, red_v_max_), h_binary_high);
        bitwise_or(h_binary_low, h_binary_high, binary_image_);
        int nonZeroCount = countNonZero(binary_image_);
        if(nonZeroCount > 30)
        {
            right_armor.color = 2;
            right_armor.hsv_roi = binary_image_.clone();
            qualifiedArmors.push_back(right_armor);
        }
    }
}

// 标准化灯条四点顺序
void standardizeBar(Bar &bar)
{
    cv::Point2f src_points[4];
    bar.rect.points(src_points);

    cv::Point2f lb, lt, rb, rt;

    std::sort(src_points, src_points + 4, [](const cv::Point2f &a, const cv::Point2f &b) {
        return a.y >= b.y;
    });

    // 下方点中 x 较小的为左下，x 较大的为右下
    if (src_points[0].x < src_points[1].x)
    {
        lb = src_points[0];
        rb = src_points[1];
    }
    else
    {
        lb = src_points[1];
        rb = src_points[0];
    }
    // 上方点中 x 较小的为左上，x 较大的为右上
    if (src_points[2].x < src_points[3].x)
    {
        lt = src_points[2];
        rt = src_points[3];
    }
    else
    {
        lt = src_points[3];
        rt = src_points[2];
    }

    // 计算灯条长度
    double norm_width = std::sqrt(std::pow(rt.x - lt.x, 2) + std::pow(rt.y - lt.y, 2));
    double norm_height = std::sqrt(std::pow(lt.x - lb.x, 2) + std::pow(lt.y - lb.y, 2));

    double long_one = std::max(norm_width, norm_height);

    bar.sorted_points.push_back(lb);
    bar.sorted_points.push_back(lt);
    bar.sorted_points.push_back(rt);
    bar.sorted_points.push_back(rb);
    bar.long_one = long_one;
}

bool deepProcess::isValidBar(Bar &bar)
{
    std::cout<<"bar.ratio: "<<bar.ratio<<"    bar.angle: "<<bar.angle<<std::endl;

    if (bar.ratio >= max_lw_ratio_ || bar.ratio <= min_lw_ratio_)
        return false;

    if (bar.angle <= min_bar_angle_ && bar.angle >= -min_bar_angle_)
        return false;

    return true;
}

// 对灯条异常情况的处理
// bool Additional_processing(std::vector<Bar> &tmpbars, cv::Point roi_center)
// {
//     // 能进入这一步的装甲板，如果是正常的，那么大概率只有两种情况：
//     // 1.存在暗部使二值化时灯条截断
//     // 2.旋转过程中下一个装甲板的灯条入镜
//     // 解决办法：
//     // 1.根据图像中心点划分左右区域
//     std::vector<Bar> left_bars;
//     std::vector<Bar> right_bars;
//     for (auto &bar : tmpbars)
//     {
//         if (bar.center.x < roi_center.x)
//         {
//             left_bars.push_back(bar);
//         }
//         else
//         {
//             right_bars.push_back(bar);
//         }
//     }
//
//     // 2.检查左右区域的灯条数量(仅处理异常数量为2的情况，两种情况同时发生之类的极端情况不予处理)
//     auto processSide = [](std::vector<Bar> &side_bars, std::string side)
//     {
//         int situation = 0;
//
//         if (side_bars.size() == 1){}
//         else if (side_bars.size() == 2)
//         {
//             // 3.根据灯条中心的x，y差距判断是哪一种情况
//             float x_diff = abs(side_bars[0].center.x - side_bars[1].center.x);
//             float y_diff = abs(side_bars[0].center.y - side_bars[1].center.y);
//
//             if (x_diff < y_diff)
//             {
//                 if (x_diff < y_diff * 0.7)
//                 {
//                     situation = 1;
//                 }
//                 else
//                 {
//                     situation = 2;
//                 }
//             }
//             else
//             {
//                 situation = 2;
//             }
//         }
//         else
//         {
//             return false;
//         }
//
//         switch (situation)
//         {
//             case 0:
//             {
//                 break;
//             }
//             case 1:
//             {
//                 // 灯条截断
//                 sort(side_bars.begin(), side_bars.end(), [](Bar &a, Bar &b) {
//                     return a.center.y < b.center.y;
//                 });
//
//                 cv::Point2f new_center((side_bars[0].center.x + side_bars[1].center.x) / 2, (side_bars[0].center.y + side_bars[1].center.y) / 2);
//
//                 // 直接取两个灯条的上下点融合
//                 Bar new_bar;
//                 new_bar.sorted_points.push_back(side_bars[1].sorted_points[0]);
//                 new_bar.sorted_points.push_back(side_bars[0].sorted_points[1]);
//                 new_bar.sorted_points.push_back(side_bars[0].sorted_points[2]);
//                 new_bar.sorted_points.push_back(side_bars[1].sorted_points[3]);
//                 new_bar.center = new_center;
//
//                 std::vector<cv::Point2f> points = new_bar.sorted_points;
//                 new_bar.rect = minAreaRect(points);
//                 float width = new_bar.rect.size.width;
//                 float height = new_bar.rect.size.height;
//                 new_bar.long_one = std::max(width, height);
//                 new_bar.ratio = new_bar.long_one / std::min(width, height);
//                 new_bar.angle = new_bar.rect.angle;
//
//                 side_bars.clear();
//                 side_bars.push_back(new_bar);
//
//                 break;
//             }
//             case 2:
//             {
//                 // 多余灯条
//                 // 直接除去最边上的灯条
//                 if (side == "left")
//                 {
//                     side_bars.erase(side_bars.begin());
//                 }
//                 else if (side == "right")
//                 {
//                     side_bars.pop_back();
//                 }
//
//                 break;
//             }
//         }
//     };
//
//     processSide(left_bars, "left");
//     processSide(right_bars, "right");
//
//     tmpbars.clear();
//     if (left_bars.size() == 1 && right_bars.size() == 1)
//     {
//         tmpbars.push_back(left_bars[0]);
//         tmpbars.push_back(right_bars[0]);
//         return true;
//     }
//     else
//     {
//         return false;
//     }
// }

// 定位灯条准确位置
void deepProcess::barFiliter(std::vector<inferredArmor> &qualifiedArmors)
{
    // 遍历每个装甲板
    for (auto &armor : qualifiedArmors)
    {
        cv::UMat roi = armor.hsv_roi;
        std::vector<Bar> tmpbars;

        cv::Mat close_kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 7));
        morphologyEx(roi, morphology_img_, cv::MORPH_CLOSE, close_kernel);

        std::vector<std::vector<cv::Point>> contours;
        findContours(morphology_img_, contours,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (auto &contour : contours)
        {
            if (contourArea(contour) < 20)
                continue;

            cv::RotatedRect rect = minAreaRect(contour);

            cv::Point2f vertices[4];
            rect.points(vertices);

            float angle;
            float width = rect.size.width;
            float height = rect.size.height;
            float max_len = std::max(width, height);
            float ratio = max_len / std::min(width, height);

            if (max_len == height)
                angle = rect.angle + 90.0f;
            else
                angle = rect.angle;

            Bar tmpbar;
            tmpbar.rect = rect;
            tmpbar.center = rect.center;
            tmpbar.ratio = ratio;
            tmpbar.angle = angle;

            if (isValidBar(tmpbar))
            {
                tmpbars.push_back(tmpbar);
            }
        }

        sort(tmpbars.begin(), tmpbars.end(), [](const Bar& a, const Bar& b) {
            return a.center.x < b.center.x;
        });

        for (auto &tmpbar : tmpbars)
        {
            standardizeBar(tmpbar);
        }

        armor.bars = tmpbars;
    }
}

bool deepProcess::isValidArmor(inferredArmor &armor)
{
    // 灯条两两匹配
    std::vector<Bar> bars = armor.bars;
    int match[2];

    auto pairArmor = [this](std::vector<Bar> &bars, int match[2])
    {
        for (int i = 0; i < bars.size() - 1; i++)
        {
            for (int j = i + 1; j < bars.size(); j++)
            {
                Bar left_bar = bars[i];
                Bar right_bar = bars[j];

                double distance = std::sqrt(std::pow(left_bar.center.x - right_bar.center.x, 2) + std::pow(left_bar.center.y - right_bar.center.y, 2));

                double bars_length = left_bar.long_one + right_bar.long_one;
                double ratio = std::max(left_bar.long_one, right_bar.long_one) / std::min(left_bar.long_one, right_bar.long_one);

                double raw_angle_diff = fabs(left_bar.angle - right_bar.angle);
                double angle = fmod(raw_angle_diff, 180);
                if (angle > 90)
                    angle = 180 - angle;

                if (distance <= bars_length * min_bars_distance_ || distance >= bars_length * max_bars_distance_)
                    continue;

                if (ratio > max_bars_ratio_)
                    continue;

                if (angle > max_bars_angle_)
                    continue;

                match[0] = i;
                match[1] = j;

                std::cout<<"match: "<<match[0]<<"    "<<match[1]<<std::endl;
                return true;
            }
        }
        return false;
    };

    if (pairArmor(bars, match))
    {
        armor.bars.clear();
        armor.bars.push_back(bars[match[0]]);
        armor.bars.push_back(bars[match[1]]);
        return true;
    }

    return false;
}

// 定位装甲板准确位置
void deepProcess::armorFiliter(std::vector<inferredArmor> &qualifiedArmors, std::vector<finalArmor> &finalArmors)
{
    for (auto &armor : qualifiedArmors)
    {
        if (armor.bars.empty())
        {
            std::cout<<"No bars found!"<<std::endl;
            continue;
        }

        if (!isValidArmor(armor))
            continue;

        std::cout<<"find armors"<<std::endl;

        std::vector<Bar> bars = armor.bars;

        // 计算装甲板四顶点
        std::vector<cv::Point2f> left(4);            // lb, lt, rt, rb
        std::vector<cv::Point2f> right(4);
        std::vector<cv::Point2f> armor_points(4);    // lt, rt, rb, lb
        cv::Point2f center;

        // 转换为全局坐标系
        for (auto &bar : bars)
        {
            for (auto &point : bar.sorted_points)
            {
                point += cv::Point2f(armor.box.x, armor.box.y);
            }

            bar.center += cv::Point2f(armor.box.x, armor.box.y);
        }

        left = bars[0].sorted_points;
        right = bars[1].sorted_points;

        armor_points[0] = (left[1] + left[2]) * 0.5;
        armor_points[1] = (right[1] + right[2]) * 0.5;
        armor_points[2] = (right[0] + right[3]) * 0.5;
        armor_points[3] = (left[0] + left[3]) * 0.5;

        // 对角线计算装甲板中心点
        double line1_k = (armor_points[1].y - armor_points[3].y) / (armor_points[1].x - armor_points[3].x + 1e-5);
        double line2_k = (armor_points[0].y - armor_points[2].y) / (armor_points[0].x - armor_points[2].x + 1e-5);
        double line1_b = armor_points[1].y - line1_k * armor_points[1].x;
        double line2_b = armor_points[0].y - line2_k * armor_points[0].x;

        double center_x = (line1_b - line2_b) / (line2_k - line1_k + 1e-5);
        double center_y = line1_k * center_x + line1_b;
        center.x = static_cast<float>(center_x);
        center.y = static_cast<float>(center_y);

        // 存储最终结果
        finalArmor final_armor;
        final_armor.armor_points = armor_points;
        final_armor.center = center;
        final_armor.color = armor.color;
        final_armor.num_roi = armor.num_roi;
        final_armor.cls_conf = armor.cls_conf;

        finalArmors.push_back(final_armor);
    }
}

// 绘制灯条
void deepProcess::show_bar(cv::Mat &img, std::vector<inferredArmor> &qualifiedArmors)
{
    for (auto &armor : qualifiedArmors)
    {
        for (auto &bar : armor.bars)
        {
            for (int i = 0; i < 4; i++)
            {
                line(img, bar.sorted_points[i], bar.sorted_points[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }
    }
}

// 绘制检测框
void deepProcess::show_box(cv::Mat &img, std::vector<finalArmor> &finalArmors)
{
    for (auto &armor : finalArmors)
    {
        for (int i = 0; i < 4; i++)
        {
            line(img, armor.armor_points[i], armor.armor_points[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        circle(img, armor.center, 3, cv::Scalar(0, 255, 0), -1);
        putText(img, armor.label, armor.center+cv::Point2f(5,5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        std::cout<<"label: "<<armor.label<<std::endl;
        std::string color = colors[armor.color];
        putText(img, color, armor.armor_points[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}

// 绘制追踪框
void deepProcess::show_track(cv::Mat &img, sensor_msgs::CameraInfoConstPtr &camera_info_, std::vector<geometry_msgs::PointStamped> all_points_, geometry_msgs::PointStamped compute_point_)
{
    // 获取相机内参矩阵
    cv::Mat_<double> camera_mat_k(3, 3, const_cast<double*>(camera_info_->K.data()));
    // 获取旋转向量
    cv::Mat rvec = cv::Mat::eye(3, 3, CV_64F);

    std::vector<cv::Point2f> image_points;
    int count{};

    // draw track
    for (auto &point : all_points_)
    {
        ++count;
        // 获取平移向量
        cv::Mat tvec_track = cv::Mat((cv::Mat_<double>(3, 1) << point.point.x, point.point.y, point.point.z));

        // 将三维点投影到图像平面
        projectPoints(std::vector{ cv::Point3f(0, 0, 0)}, rvec, tvec_track,
                      camera_mat_k, camera_info_->D, image_points);

        if (count != all_points_.size())
        {
            // armor
            for (auto point : image_points)
            {
                circle(img, point, 10, cv::Scalar(0, 255, 0), -1);
            }
        }
        else
        {
            // center
            for (auto point : image_points)
            {
                circle(img, point, 10, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    // draw compute
    // 获取平移向量
    cv::Mat tvec_compute = cv::Mat((cv::Mat_<double>(3, 1) << compute_point_.point.x, compute_point_.point.y, compute_point_.point.z));

    projectPoints(std::vector{ cv::Point3f(0, 0, 0)}, rvec, tvec_compute,
                  camera_mat_k, camera_info_->D, image_points);

    for (auto point : image_points)
    {
        circle(img, point, 10, cv::Scalar(0, 255, 0), -1);
    }
}