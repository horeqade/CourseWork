#include <stdio.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/eval.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <string>
#include <direct.h>



namespace py = pybind11;
using namespace pybind11::literals;

std::string GetCurrentWorkingDir(void) {
	char buff[FILENAME_MAX];
	_getcwd(buff, FILENAME_MAX);
	std::string path(buff);
	path = path.substr(0, size(path) - 5);
	return path;
}


class generator {
 public:
  generator(std::string json_path, std::string weights_path,
            bool abs_path_json = false, bool abs_path_weights = false) {
    // abs_path_weigths � abs_path_json ������ ���� true, 
    //���� ������� ���������� ���� �������������� ������
    py::object json;
    if (abs_path_json) { //������ json ����� ��������� �������
      json = open(json_path);
    } else {
      json = open(default_path + json_path);
    }
    model = model_from_json(json.attr("read")().cast<std::string>()); //�������� ������ � json
    if (abs_path_weights) {        //�������� ����� � ������
      model.attr("load_weights")(weights_path);
    } else {
      model.attr("load_weights")(default_path + weights_path);
    }    
  }
  py::array_t<double> make_pics(int pic_num = 1) { //pic_num ���������� �������� ��� ������
    py::object dims = util.attr("mkarr")(pic_num, 100);    //�������� ��������� � ��������� ������������
    py::object z = np_random.attr("normal")(0, 1, dims);  //��������� ����� ��� ����� � ������
    py::array_t<double> pics_arr = model.attr("predict")(z);    //��������� �������� � np.array
    util.attr("save")(pics_arr, default_path);        //���������� ��������
    return pics_arr;
  }

 private:
  py::object util = py::module::import("utilities");
  py::object model_from_json =
            py::module::import("keras.models").attr("model_from_json");
  py::object np_random = py::module::import("numpy.random");
  py::object cv2 = py::module::import("cv2");
  py::object model;
  std::string default_path{GetCurrentWorkingDir() + "\\"};
  std::string json_path{""};
  std::string weights_path{""};
  py::object open(std::string path) {
    return py::module::import("builtins").attr("open")(path, "r");
  }
};

class rcnn {
public:
  rcnn(std::string weights_path) {
    py::object sys_path = py::module::import("sys").attr("path");
    sys_path.attr("append")(root_dir);
    sys_path.attr("append")(root_dir + "\\samples\\coco\\"); //��������� ���� ��� ������� ���������
    py::object coco = py::module::import("coco");
    py::object model = modellib.attr("MaskRCNN")("mode"_a = "inference", "model_dir"_a = model_dir, 
	      "config"_a = coco.attr("CocoConfig")());   //�������� ������
    model.attr("load_weights")(weights_path, "by_name"_a = true);  //�������� �����
	}
  py::array_t<double> detect() {
    py::object imread = py::module::import("skimage.io").attr("imread");  
    py::object img = imread(image_path);             //�������� �����������
	  py::list list_img;
	  list_img.append(img);
    py::array_t<double> results = model.attr("detect")("images"_a = list_img, "verbose"_a = 1);
    std::cout << "Detected !" << std::endl;
    util.attr("save_detected")(results, root_dir + "\\detected.png");
    return results;
	}
private:
	py::object model;
	std::string root_dir{GetCurrentWorkingDir() + "\\Mask_RCNN-master"};
	py::object modellib{ py::module::import("mrcnn.model") };
	std::string model_dir = root_dir + "\\logs";
	std::string coco_model_path = root_dir + "\\mask_rcnn_coco.h5";
	std::string image_path = root_dir + "\\image.jpg";
	py::object util = py::module::import("utilities");
};

class linreg {
public:
  linreg(int k, int b) {
    py::object SimpleNetwork = py::module::import("SimpleNetwork").attr("SimpleNetwork");
    model = SimpleNetwork(k,b);         //�������� ������ � ��������� ���������� k � b
  }
  void train(int num_steps) {
    model.attr("train")(num_steps);    
  }
  void MakeGif(std::string gif_path) {
    model.attr("MakeGif")(gif_path);      //������� ����� � � ������ ������
  }
private:
  py::object model;
};

class srgan {
public:
  srgan() {}
  py::array_t<double> upscale(std::string weights_path) {
    py::object sys_path = py::module::import("sys").attr("path");
    sys_path.attr("append")(root_dir);
    py::object main = py::module::import("main");
    py::array_t<double> result = main.attr("evaluate")(root_dir, weights_path);
    //����� ��������� ������, ����, ����������� �������� � ���������

    return result;

  }
private:
  std::string root_dir = GetCurrentWorkingDir() + "\\srgan-master";
};

class yolo {
public:
  yolo() {}
  py::array_t<double> detect() {
    py::object sys_path = py::module::import("sys").attr("path");
    sys_path.attr("append")(root_dir);
    py::object YoloDetect = py::module::import("YoloDetect");
    py::array_t<double> img_arr = YoloDetect.attr("main")(root_dir);                   //�������� ������ � �����, ���������� �������� � �������
    return img_arr;
  }
private:
  std::string root_dir = GetCurrentWorkingDir() + "\\yolo";
};

int main() {

  //������� ���� �� ������������ ������������ Python
  Py_SetPythonHome(L"D:/programming/Anaconda1/envs/Coursach");

  py::scoped_interpreter python;
  py::object sys_path = py::module::import("sys").attr("path");
  sys_path.attr("append")(GetCurrentWorkingDir() + "\\");

  //������� ������������� �������:
  
  //generator gen("23.12 gen_model.json", "12.12 830_epoch gen_weights.h5");
  //result = gen.make_pics(4);

  //rcnn detector;
  //detector.detect();

  //linreg model(15, -15);
  //model.train(500);
  //model.MakeGif((GetCurrentWorkingDir() + "\\lin_reg.gif"));

  //srgan supres;
  //result = supres.upscale();

  //yolo detector;
  //detector.detect();

  std::cout << GetCurrentWorkingDir() << std::endl;
  return 0;
}