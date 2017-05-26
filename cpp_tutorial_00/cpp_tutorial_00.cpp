/* File: cpp_tutorial_00.cpp
*
Copyright (c) [2016] [Mohammad Hosseinabady (mohammad@hosseinabady.com)]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================
* This file has been written at University of Bristol
* for the ENPOWER project funded by EPSRC
*
* File name : cpp_tutorial_00.cpp
* author    : Mohammad hosseinabady mohammad@hosseinabady.com
* date      : 1 October 2016
* blog: https://highlevel-synthesis.com/
*/




//Step 00—Include required header files
#include <caffe/caffe.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace caffe;  




int main(int argc, char** argv) {

// 01 -- Select CPU or GPU
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

// 02 -- Define network
	shared_ptr<Net<float> > net_;
	int num_channels_;
	string model_file   = argv[1];

// 03 -- Load the network from file 
	net_.reset(new Net<float>(model_file, TEST));

// 04 -- assign weights  
	shared_ptr<Layer<float> > conv_layer = net_->layer_by_name("conv");
	float* weights = conv_layer->blobs()[0]->mutable_cpu_data();


	weights[0] = -1;	weights[1] = 0; 	weights[2] = 1;
	weights[3] = -2;	weights[4] = 0;  	weights[5] = 2;
	weights[6] = -1;	weights[7] = 0; 	weights[8] = 1;


// 05 -- load the input image  
	string image_file = argv[2];
	cv::Mat img = cv::imread(image_file, -1);

// 06: reshape the input blob to the size of the input image
	shared_ptr<Blob<float> > input_blob = net_->blob_by_name("data");
	num_channels_ = input_blob->channels();
	input_blob->Reshape(1, num_channels_, img.rows, img.cols);

	
// 07: reshape the whole network correspondingly
	net_->Reshape();


// 08: copy the input image to the network input blob
	int width = input_blob->width();
	int height = input_blob->height();
	float* input_data = input_blob->mutable_cpu_data();
	cv::Mat channel(height, width, CV_32FC1, input_data);
	img.convertTo(channel, CV_32FC1);


// 09: run the NN inference
	net_->Forward();
	
//10: get the output and save in a file  
	Blob<float>* output_layer = net_->output_blobs()[0];
	int num_out_channels_ = output_layer->channels();

	width = output_layer->width();
	height = output_layer->height();
	float* output_data = output_layer->mutable_cpu_data();
	cv::Mat outputImage(height, width, CV_32FC1, output_data);

	imwrite("C:/tmp/outout_Image.jpg", outputImage);


	return 0;

}






