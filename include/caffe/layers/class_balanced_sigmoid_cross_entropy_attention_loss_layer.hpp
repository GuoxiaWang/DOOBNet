#ifndef CAFFE_CLASS_BALANCED_SIGMOID_CROSS_ENTROPY_ATTENTION_LOSS_LAYER_HPP_
#define CAFFE_CLASS_BALANCED_SIGMOID_CROSS_ENTROPY_ATTENTION_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

//By Guoxia Wang 2018/03/15
template <typename Dtype>
class ClassBalancedSigmoidCrossEntropyAttentionLossLayer : public LossLayer<Dtype> {
 public:
  explicit ClassBalancedSigmoidCrossEntropyAttentionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_output_(new Blob<Dtype>()) {
		  LayerParameter param_t = param;
		  param_t.clear_loss_weight();
		  sigmoid_layer_.reset(new SigmoidLayer<Dtype>(param_t));
	  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ClassBalancedSigmoidCrossEntropyAttentionLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }

 protected:
  /// @copydoc ClassBalancedSigmoidCrossEntropyAttentionLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual Dtype get_normalizer(
	  LossParameter_NormalizationMode normalization_mode, int valid_count);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
  Dtype alpha_, gamma_, valid_num_, beta_;
  Blob<Dtype> scaler_;
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
  Dtype normalizer_;
  int outer_num_, inner_num_;


};

}  // namespace caffe

#endif  // CAFFE_CLASS_BALANCED_SIGMOID_CROSS_ENTROPY_ATTENTION_LOSS_LAYER_HPP_
