#include <inference.h>

//#define number_classifier_model_path_ "/home/lovod/rm_code/src/rm_visplugin/rm_digitalimg_proc_deep/model/mlp_dx.onnx"
#define number_classifier_model_path_ "/home/lovod/rm_code/src/rm_visplugin/rm_digitalimg_proc_deep/model/number_classifier.onnx"
#define roi_w_ 20
#define roi_h_ 28
#define light_len_    12    // 灯条长度
#define warp_height_  28    // 透视变换后图像高度
#define warp_width_   32    // 透视变换后图像宽度(即小装甲板宽度)

static const std::vector<std::string> class_names_ = {
    "1", "2", "3", "4", "5", "outpost", "guard", "base", "negative"};

class number_classifier {
public:
    number_classifier()
    {
        using_once();
    }
    void using_once();
    void init();

    void warp(finalArmor &armors);
    cv::UMat extractNumbers(finalArmor &armors);
    void classify(std::vector<finalArmor> &finalArmors);

private:
    std::once_flag flag_;
    cv::dnn::Net net_;
};