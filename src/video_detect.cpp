// #include <inference.h>
//
// int main()
// {
//     deepProcess deep_process;
//
//     cv::VideoCapture cap("/home/lovod/rm_code/src/rm_visplugin/rm_digitalimg_proc_deep/test/test.mp4");
//
//     while (true)
//     {
//         cv::UMat img;
//         cap >> img;
//
//         if (img.empty())
//         {
//             break;
//         }
//
//         dataImg img_data = deep_process.preprocess_img(img);
//
//         std::vector<finalArmor> finalArmors;
//         finalArmors = deep_process.InferAndPostprocess(img_data);
//         deep_process.show_box(img, finalArmors);
//         imshow("result", img);
//         cv::waitKey(0);
//     }
//
//     return 0;
// }