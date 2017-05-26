/*
 * simple_convolution.cpp
 *
 *  Created on: 11 May 2017
 *      Author: mohammad
 */


#include <caffe/caffe.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace caffe;  // NOLINT(build/namespaces)
using std::string;



int main(int argc, char** argv) {

/*01 -- Select CPU or GPU*/
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

/*02 -- Define network*/
	shared_ptr<Net<float> > net_;
	int num_channels_;
	string model_file   = argv[1];

/* 03 -- Load the network from file */
	net_.reset(new Net<float>(model_file, TEST));

/* 04 -- assign weights  */
	shared_ptr<Layer<float> > sobel_layer = net_->layer_by_name("conv");
	float* weights = sobel_layer->blobs()[0]->mutable_cpu_data();


	weights[0] = -1;	weights[1] = 0; 	weights[2] = 1;
	weights[3] = -2;	weights[4] = 0;  	weights[5] = 2;
	weights[6] = -1;	weights[7] = 0; 	weights[8] = 1;


/* 05 -- read the input image  */
	string image_file = argv[2];
	cv::Mat img = cv::imread(image_file, -1);

	shared_ptr<Blob<float> > input_blob = net_->blob_by_name("data");
	num_channels_ = input_blob->channels();
	input_blob->Reshape(1, num_channels_, img.rows, img.cols);

	/* Forward dimension change to all layers. */
	net_->Reshape();



	int width = input_blob->width();
	int height = input_blob->height();
	float* input_data = input_blob->mutable_cpu_data();
	cv::Mat channel(height, width, CV_32FC1, input_data);
	img.convertTo(channel, CV_32FC1);


	net_->Forward();
	
  
	Blob<float>* output_layer = net_->output_blobs()[0];
	int num_out_channels_ = output_layer->channels();

	
	width = output_layer->width();
	height = output_layer->height();
	float* output_data = output_layer->mutable_cpu_data();
	cv::Mat outputImage(height, width, CV_32FC1, output_data);


	imwrite("C:/tmp/outout_Image.jpg", outputImage);

	return 0;

}






