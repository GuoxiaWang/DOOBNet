// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified Guoxia Wang
// ------------------------------------------------------------------

#include <vector>

#include "caffe/layers/orientation_smooth_L1_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void	CalcuOriDiff(const int n, const Dtype* input_data, const Dtype* target, Dtype* diff){
	CUDA_KERNEL_LOOP(index, n) {
		Dtype input_val = input_data[index];
		Dtype target_val = target[index];
		const Dtype PI = Dtype(3.141592654);
		if ((input_val > PI && target_val > Dtype(0))
			|| (input_val < -PI && target_val < Dtype(0))){
			diff[index] = input_val + target_val;
		} else {
			diff[index] = input_val - target_val;
		}
	}
}

template <typename Dtype>
__global__ void SmoothL1Forward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
  //        |x| - 0.5 / sigma / sigma    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = 0.5 * val * val * sigma2;
    } else {
      out[index] = abs_val - 0.5 / sigma2;
    }
  }
}

template <typename Dtype>
void OrientationSmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  // d := b0 - b1
  // caffe_gpu_sub(
  //   count,
  //   bottom[0]->gpu_data(),
  //   bottom[1]->gpu_data(),
  //   diff_.mutable_gpu_data());    // d := b0 - b1

  CalcuOriDiff<Dtype><< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>> >(
      count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), diff_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  // apply weights
  caffe_gpu_mul(
      count,
      bottom[2]->gpu_data(),
      diff_.gpu_data(),
      diff_.mutable_gpu_data());  // d := w_in * (b0 - b1)

  SmoothL1Forward<Dtype><< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>> >(
      count, diff_.gpu_data(), errors_.mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;

  Dtype loss;
  caffe_gpu_dot(count, ones_.gpu_data(), errors_.gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
__global__ void SmoothL1Backward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
  //       = sign(x)                   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = sigma2 * val;
    } else {
      out[index] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
}

template <typename Dtype>
void OrientationSmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  if ( propagate_down[ 1 ]) {
	  LOG(FATAL) << this->type()
		  << " Layer cannot backpropagate to label inputs.";
  }
  if ( propagate_down [2]) {
	  LOG(FATAL) << this->type()
		  << " Layer cannot backpropagate to mask inputs.";
  }  
  if ( propagate_down[ 0 ] ) {
  	  // after forwards, diff_ holds w_in * (b0 - b1)
	  int count = diff_.count();
	  SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	      count, diff_.gpu_data(), diff_.mutable_gpu_data(), sigma2_);
	  CUDA_POST_KERNEL_CHECK;

	  const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
	  caffe_gpu_axpby(
	      count,                           // count
	      alpha,                           // alpha
	      diff_.gpu_data(),                // x
	      Dtype(0),                        // beta
	      bottom[0]->mutable_gpu_diff());  // y

	  // Scale by weight
	  caffe_gpu_mul(
		  count,
		  bottom[2]->gpu_data(),
		  bottom[0]->gpu_diff(),
		  bottom[0]->mutable_gpu_diff());
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(OrientationSmoothL1LossLayer);

}  // namespace caffe
