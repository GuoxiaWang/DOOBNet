#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_labelmap_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageLabelmapDataLayer<Dtype>::~ImageLabelmapDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageLabelmapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  string data_type = this->layer_param_.image_data_param().data_type(); // for selecting the label data format 

  //LOG(INFO)<<data_type; 

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string img_filename;
  string gt_filename;
  while (infile >> img_filename >> gt_filename) {
    lines_.push_back(std::make_pair(img_filename, gt_filename));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  //const int channels = cv_img.channels(); 
  const int height = cv_img.rows; 
  const int width = cv_img.cols; 
  int gt_height, gt_width; 

  vector<int> top_shape_labelmap ;

  if(data_type.compare("img") == 0)
  {
      cv::Mat cv_gt = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                                    new_height, new_width, 0);
      const int gt_channels = cv_gt.channels(); 
      gt_height = cv_gt.rows; 
      gt_width = cv_gt.cols; 
      
      CHECK(gt_channels == 1) << "GT image channel number should be 1";
      if (new_height > 0 && new_width > 0)
        cv::resize(cv_gt, cv_gt, cv::Size(new_width, new_height));
      top_shape_labelmap = this->data_transformer_->InferBlobShape(cv_gt);
  }
  else if(data_type.compare("h5") == 0)
  {   
      // using hdf5 file to load blob
      Blob<Dtype> gt_blob; 
      string hdfname = root_folder+lines_[lines_id_].second; 
      LoadHDF5FileToBlob(hdfname.c_str(), string("label"), gt_blob); 
      
      gt_height = gt_blob.height(); 
      gt_width = gt_blob.width(); 

      top_shape_labelmap = this->data_transformer_->InferBlobShape(&gt_blob);
      //CHECK((new_height==0) || (new_height == 0)) <<"current not support resize the hdf5 data"; 
  }
  else
  {
    CHECK(0)<<"no such data type for input"; 
  }

  CHECK((height == gt_height) && (width == gt_width)) << "groundtruth size != image size"; 

  if (new_height > 0 && new_width > 0) {
    cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
  }

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  
  
  this->transformed_data_.Reshape(top_shape);
  this->transformed_labelmap_.Reshape(top_shape_labelmap);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  // if (new_height == 0 || new_width == 0) {
  //   CHECK_EQ(batch_size, 1) << "if size of image is adaptively changing, must set batch to 1";
  // }

  top_shape[0] = batch_size;
  top_shape_labelmap[0] = batch_size; // change to batch_size 
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
    this->prefetch_[i].labelmap_.Reshape(top_shape_labelmap);
  }
  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape_labelmap);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  LOG(INFO) << "output label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
}

template <typename Dtype>
void ImageLabelmapDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageLabelmapDataLayer<Dtype>::load_batch(LabelmapBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(batch->labelmap_.count());
  CHECK(this->transformed_data_.count());
  CHECK(this->transformed_labelmap_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  const Dtype gt_edge_sampling_threshold = image_data_param.gt_edge_sampling_threshold();
  string root_folder = image_data_param.root_folder();
  string data_type = image_data_param.data_type(); 

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  vector<int> top_shape_labelmap;
  //LOD(INFO)<<data_type; 

  if(data_type.compare("img") == 0)
  {
    cv::Mat cv_gt = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
      new_height, new_width, 0);
    top_shape_labelmap = this->data_transformer_->InferBlobShape(cv_gt);
  }
  else if (data_type.compare("h5") == 0)
  {
    Blob<Dtype> gt_blob;
    string hdfname = root_folder + lines_[lines_id_].second; 
    LoadHDF5FileToBlob(hdfname.c_str(), string("label"), gt_blob); 
    top_shape_labelmap = this->data_transformer_->InferBlobShape(&gt_blob); 
  }

  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);

  this->transformed_data_.Reshape(top_shape);
  this->transformed_labelmap_.Reshape(top_shape_labelmap);
  // Reshape prefetch_data and top[0] according to the batch_size.
  top_shape[0] = batch_size;
  top_shape_labelmap[0] = batch_size;
  
  batch->data_.Reshape(top_shape);
  batch->labelmap_.Reshape(top_shape_labelmap);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_labelmap = batch->labelmap_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    0, 0, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

    if (!cv_img.data) {
      continue;
    }

    Blob<Dtype> gt_blob; 
    cv::Mat cv_gt;
    int gt_height, gt_width; 

    if(data_type.compare("img")==0)
    {
      cv_gt = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                                    0, 0, 0);
      if(!cv_gt.data)
        continue; 

      gt_height = cv_gt.rows;
      gt_width = cv_gt.cols;
      if (new_height > 0 && new_width > 0)
        cv::resize(cv_gt, cv_gt, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    }

    else if (data_type.compare("h5")==0)
    {
      // load the ground truth data 
      string hdfname = root_folder + lines_[lines_id_].second; 
      LoadHDF5FileToBlob(hdfname.c_str(), string("label"), gt_blob); 
      gt_height = gt_blob.height(); 
      gt_width = gt_blob.width(); 
    }
    
    const int height = cv_img.rows;
    const int width = cv_img.cols;

    CHECK((height == gt_height) && (width == gt_width)) << "GT image size should be equal to true image size";
    
    if (new_height > 0 && new_width > 0) {
        cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
    }

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    int offset_gt = batch->labelmap_.offset(item_id);

    //CHECK(offset == offset_gt) << "fetching should be synchronized";
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->transformed_labelmap_.set_cpu_data(prefetch_labelmap + offset_gt);
    int h_off = 0;
    int w_off = 0;
    bool do_mirror = false;

    this->data_transformer_->LocTransform(cv_img, &(this->transformed_data_), h_off, w_off, do_mirror);


    if(data_type.compare("img")==0)
    {
      //regression
      //[***Cautions***]
      //One small trick leveraging opencv roundoff feature for **consensus sampling** in Holistically-Nested Edge Detection paper.
      //For general binary edge maps this is okay
      //For 5-subject aggregated edge maps (BSDS), this will abandon weak edge points labeled by only two or less labelers.

      //[***ChangeLog***]
      //Guoxia Wang
      //Add a threshold variable to **sonsensus sampling** 
      cv::Mat encoded_gt;
      cv_gt.convertTo(cv_gt, CV_32F);
      encoded_gt = cv_gt/255;
      this->data_transformer_->LabelmapTransform(encoded_gt, &(this->transformed_labelmap_), h_off, w_off, do_mirror, gt_edge_sampling_threshold);
    }
    else if(data_type.compare("h5")==0)
    {
      this->data_transformer_->LabelTransform(&(gt_blob), &(this->transformed_labelmap_), h_off, w_off, do_mirror); 
    }
    else
    {
      CHECK(0)<<"No such data type for input label"; 
    }

    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageLabelmapDataLayer);
REGISTER_LAYER_CLASS(ImageLabelmapData);

}  // namespace caffe
#endif  // USE_OPENCV
