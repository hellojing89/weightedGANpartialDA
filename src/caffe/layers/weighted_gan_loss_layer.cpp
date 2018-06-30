#include <algorithm>
#include <vector>

#include "caffe/layers/weighted_gan_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedGanLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.loss_param().has_normalization()) {
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
}

template <typename Dtype>
void WeightedGanLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "WEIGHTED_GAN_LOSS layer inputs must have the same count.";
  CHECK_EQ(bottom[2]->count(), bottom[1]->count()) <<
      "WEIGHTED_GAN_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  weights_.ReshapeLike(*bottom[2]);
}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
template <typename Dtype>
Dtype WeightedGanLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void WeightedGanLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs, which are used for backward
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* input_weight = bottom[2]->cpu_data();
  Dtype* diff_l = weights_.mutable_cpu_diff(); //used as buffer for backward
  Dtype* weights = weights_.mutable_cpu_data(); //used as buffer for backward
  int valid_count = 0;
  int source_count = 0;
  Dtype loss = 0;
  total_weight_ = 0;
  const Dtype one = 1;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      continue;
    }
    diff_l[i] = (input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
    if (target[i] == 1) {
        weights[i] = exp(-(input_weight[i] >= 0) * input_weight[i]) / 
            (1 + exp(input_weight[i] - 2 * input_weight[i] * (input_weight[i] >= 0)));
        total_weight_ += weights[i];
        ++source_count;
    }
    ++valid_count;
  }
  total_weight_ /= source_count;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      continue;
    }
    weights[i] = ((target[i] == 1) ? (weights[i]/total_weight_) : one);
    loss -= diff_l[i] * weights[i];
  }

  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void WeightedGanLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    const Dtype* weights = weights_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();      
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    caffe_mul(count, bottom_diff, weights, bottom_diff);
    for (int i = 0; i < count; ++i) {
      // Zero out gradient of ignored targets.
      if (has_ignore_label_) {
        const int target_value = static_cast<int>(target[i]);
        if (target_value == ignore_label_) {
          bottom_diff[i] = 0;
        }
      }
    }
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedGanLossLayer);
#endif

INSTANTIATE_CLASS(WeightedGanLossLayer);
REGISTER_LAYER_CLASS(WeightedGanLoss);

}  // namespace caffe
