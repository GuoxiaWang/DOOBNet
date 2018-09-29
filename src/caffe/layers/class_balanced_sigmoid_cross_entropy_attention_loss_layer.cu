#include <vector>

#include "caffe/layers/class_balanced_sigmoid_cross_entropy_attention_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SampleCountGPU(const int nthreads, const Dtype* target, Dtype* count_pos, Dtype* count_neg,
	const bool has_ignore_label_, const int ignore_label_) {
	CUDA_KERNEL_LOOP(i, nthreads) {
		const int target_value = static_cast<int>( target[ i ] );
		if ( has_ignore_label_ && target_value == ignore_label_ ) {
			count_pos[ i ] = 0;
			count_neg[ i ] = 0;
		}
		else {
			if (target_value == 1) {
				count_pos[ i ] = 1;
				count_neg[ i] = 0;
			}
			else {
				count_pos[ i ] = 0;
				count_neg[ i ] = 1;
			}
		}
	}
}

template <typename Dtype>
__global__ void AttentionLossForwardGPU(const int nthreads,
	const Dtype* input_data, const Dtype* sigmoid_data, const Dtype* target, Dtype* scale, Dtype* oriloss,
	const bool has_ignore_label_, const int ignore_label_,
	Dtype* counts, float alpha, float beta, float gamma) {
	CUDA_KERNEL_LOOP(i, nthreads) {
		const int target_value = static_cast<int>( target[ i ] );
		if ( has_ignore_label_ && target_value == ignore_label_ ) {
			scale[ i ] = 0;
			oriloss[ i ] = 0;
			counts[ i ] = 0;
		}
		else {
			scale[ i ] = (target_value == 1 ? alpha : 1 - alpha) * powf(beta, powf(fmaxf(1 - ( target_value == 1 ? sigmoid_data[ i ] : ( 1 - sigmoid_data[ i ] ) ), Dtype(0.00000001)), gamma ) );

			oriloss[ i ] = -(input_data[ i ] * ( target[ i ] - ( input_data[ i ] >= 0 ) ) -
				log(1 + exp(input_data[ i ] - 2 * input_data[ i ] *
				( input_data[ i ] >= 0 ))));
			counts[ i ] = 1;
		}
	}
}

template <typename Dtype>
__global__ void AttentionLossBackwardGPU(const int nthreads, const Dtype* sigmoid_data, const Dtype* target, 
	const Dtype* scale, const Dtype* oriloss, float beta, float gamma, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(i, nthreads) {
		const int target_value = static_cast<int>( target[ i ] );
		Dtype tmp = ( target_value == 1 ? -1 : 1 ) * (target_value == 1 ? sigmoid_data[ i ] : ( 1 - sigmoid_data[ i ] ) * oriloss[i] 
				* log(beta) * gamma * powf(fmaxf(target_value == 1 ? (1 - sigmoid_data[ i ]) : sigmoid_data[ i ], Dtype(0.00000001)), gamma - 1)  + 1);
		bottom_diff[i] = scale[i] * (target_value == 1 ? (1 - sigmoid_data[ i ]) : sigmoid_data[ i ]) * tmp;
	}
}

template <typename Dtype>
__global__ void AttentionLossIgnoreDiffGPU(const int count,
	const int ignore_label, const Dtype* target, Dtype* diff) {
	CUDA_KERNEL_LOOP(i, count) {
		const int target_value = static_cast<int>( target[ i ] );
		if ( target_value == ignore_label ) {
			diff[ i ] = 0;
		}
	}
}

template <typename Dtype>
void ClassBalancedSigmoidCrossEntropyAttentionLossLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// The forward pass computes the sigmoid outputs.
	sigmoid_bottom_vec_[ 0 ] = bottom[ 0 ];
	sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
	// Compute the loss (negative log likelihood)
	const int count = bottom[ 0 ]->count();
	// Stable version of loss computation from input data
	const Dtype* input_data = bottom[ 0 ]->gpu_data();
	const Dtype* target = bottom[ 1 ]->gpu_data();
	const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
	// Since this memory is not used for anything until it is overwritten
	// on the backward pass, we use it here to avoid having to allocate new GPU
	// memory to accumulate intermediate results in the kernel.
	Dtype* count_pos = bottom[ 0 ]->mutable_gpu_diff();
	Dtype* count_neg = bottom[ 1 ]->mutable_gpu_diff();
	Dtype valid_pos_count;
	Dtype valid_neg_count;
	// NOLINT_NEXT_LINE(whitespace/operators)
	SampleCountGPU<Dtype> << <CAFFE_GET_BLOCKS(count),
		CAFFE_CUDA_NUM_THREADS >> >( count, target, count_pos, count_neg, 
		has_ignore_label_, ignore_label_);
	caffe_gpu_asum(count, count_pos, &valid_pos_count);
	caffe_gpu_asum(count, count_neg, &valid_neg_count);
	alpha_ = valid_neg_count / fmaxf(valid_pos_count + valid_neg_count, 1e-5);

	Dtype* loss_data = bottom[ 0 ]->mutable_gpu_diff();
	Dtype* count_data = bottom[ 1 ]->mutable_gpu_diff();
	Dtype valid_count;
	// NOLINT_NEXT_LINE(whitespace/operators)
	AttentionLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count),
		CAFFE_CUDA_NUM_THREADS >> >( count, input_data, sigmoid_output_data, 
		target, scaler_.mutable_gpu_data(), scaler_.mutable_gpu_diff(),
		has_ignore_label_, ignore_label_, count_data, alpha_, beta_, gamma_ );
	caffe_gpu_mul(count, scaler_.gpu_data(), scaler_.gpu_diff() , loss_data);
	// Only launch another CUDA kernel if we actually need the valid count.
	if ( normalization_ == LossParameter_NormalizationMode_VALID &&
		has_ignore_label_ ) {
		caffe_gpu_asum(count, count_data, &valid_count);
	}
	else {
		valid_count = count;
	}
	Dtype loss;
	caffe_gpu_asum(count, loss_data, &loss);
	normalizer_ = get_normalizer(normalization_, valid_count);
	top[ 0 ]->mutable_cpu_data()[ 0 ] = loss / normalizer_;
	caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
	caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
}

template <typename Dtype>
void ClassBalancedSigmoidCrossEntropyAttentionLossLayer<Dtype>::Backward_gpu(
	const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
	// scale_.data := scale    .diff := oriloss
	if ( propagate_down[ 1 ] ) {
		LOG(FATAL) << this->type()
			<< " Layer cannot backpropagate to label inputs.";
	}
	if ( propagate_down[ 0 ] ) {
		// First, compute the diff
		const int count = bottom[ 0 ]->count();
		const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
		const Dtype* target = bottom[ 1 ]->gpu_data();
		const Dtype* input_data = bottom[ 0 ]->gpu_data();
		Dtype* bottom_diff = bottom[ 0 ]->mutable_gpu_diff();

		AttentionLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >( count, sigmoid_output_data, target, scaler_.gpu_data(), 
				scaler_.gpu_diff(), beta_, gamma_, bottom_diff);

		// Zero out gradient of ignored targets.
		if ( has_ignore_label_ ) {
			// NOLINT_NEXT_LINE(whitespace/operators)
			AttentionLossIgnoreDiffGPU<Dtype> << <CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS >> >( count, ignore_label_, target, bottom_diff );
		}
		// Scale down gradient
		Dtype loss_weight = top[ 0 ]->cpu_diff()[ 0 ] / normalizer_;
		caffe_gpu_scal(count, loss_weight, bottom_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ClassBalancedSigmoidCrossEntropyAttentionLossLayer);


}  // namespace caffe
