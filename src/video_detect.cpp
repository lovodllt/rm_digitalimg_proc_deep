// #include <inference.h>
// #include <number_classifier.h>
//
// int main()
// {
//     deepProcess deep_process;
//     number_classifier classifier;
//
//     cv::VideoCapture cap("/home/lovod/rm_code/src/rm_visplugin/rm_digitalimg_proc_deep/test/test3.mp4");
//     std::vector<finalArmor> finalArmors;
//     std::vector<inferredArmor> qualifiedArmors;
//     TargetColor target_color = TargetColor::BLUE;
//
//     while (true)
//     {
//         cv::Mat img;
//         cap >> img;
//
//         if (img.empty())
//         {
//             std::cout << "视频读取完毕" << std::endl;
//             break;
//         }
//
//         dataImg img_data = deep_process.preprocess_img(img);
//
//         finalArmors = deep_process.InferAndPostprocess(img_data,0.3,0.3,target_color);
//         classifier.classify(finalArmors);
//         deep_process.show_box(img, finalArmors);
//         imshow("result", img);
//         cv::waitKey(0);
//     }
//
//     return 0;
// }