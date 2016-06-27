#include <cmath>
#include <cfloat>
#include <fstream>
#include <algorithm>
#include "caffe/layers/ordinal_regression_loss_layer.hpp"

namespace caffe {

template<typename Dtype>
void OrdinalRegressionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // k
  bool has_k = this->layer_param_.ordinal_regression_loss_param().has_k();
  if (has_k) {
    k_ = this->layer_param_.ordinal_regression_loss_param().k();
  }
  else {
    k_ = bottom[0]->shape(1) / 2;
  }
  // weight for every label
  vector<int> shape(1, k_);
  weight_.Reshape(shape);
  Dtype* weight_data = weight_.mutable_cpu_data();
  bool has_weight_file = this->layer_param_.ordinal_regression_loss_param().has_weight_file();
  if (has_weight_file) {
    string weight_file = this->layer_param_.ordinal_regression_loss_param().weight_file();
    std::ifstream fin;
    fin.open(weight_file.c_str());
    Dtype weight;
    for (int i = 0; i < k_; i++) {
      fin >> weight;
      weight_data[i] = weight;
    }
    fin.close();
  }
  else {
    for (int i = 0; i < k_; i++) {
      weight_data[i] = 1;
    }
  }
}

template<typename Dtype>
void OrdinalRegressionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), 2 * k_) << "Input must be (?, 2K) dimension.";
  prob_.ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void OrdinalRegressionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int n = bottom[0]->shape(0);
  const int m = 2 * k_;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* prob_data = prob_.mutable_cpu_data();
  // get prob
  for (int i = 0; i < n; i++) {
    const int offset = bottom[0]->offset(i);
    const Dtype* x = bottom_data + offset;
    Dtype* y = prob_data + offset;
    for (int j = 0; j < m; j+=2) {
      const Dtype max_input = std::max(x[j], x[j+1]);
      y[j] = std::exp(x[j] - max_input);
      y[j+1] = std::exp(x[j+1] - max_input);
      Dtype sum = y[j] + y[j+1];
      y[j] /= sum;
      y[j+1] /= sum;
    }
  }
  // calc loss
  Dtype loss = 0;
  const Dtype* weight_data = weight_.cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  for (int i = 0; i < n; i++) {
    const Dtype* y = prob_data + prob_.offset(i);
    const int label = static_cast<int>(label_data[i]);
    for (int j = 0; j < label; j++) {
      loss -= weight_data[j] * std::log(std::max(y[2*j+1], Dtype(FLT_MIN)));
    }
    for (int j = label; j < k_; j++) {
      loss -= weight_data[j] * std::log(std::max(y[2*j], Dtype(FLT_MIN)));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / n;
}

template<typename Dtype>
void OrdinalRegressionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int n = bottom[0]->shape(0);
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
    const Dtype* weight_data = weight_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    for (int i = 0; i < n; i++) {
      const int offset = bottom[0]->offset(i);
      Dtype* dx = bottom_diff + offset;
      const int label = static_cast<int>(label_data[i]);
      for (int j = 0; j < label; j++) {
        dx[2*j+1] -= 1;
      }
      for (int j = label; j < k_; j++) {
        dx[2*j] -= 1;
      }
      for (int j = 0; j < k_; j++) {
        dx[2*j] *= weight_data[j];
        dx[2*j+1] *= weight_data[j];
      }
    }
    const Dtype scale = 1.0 / n;
    caffe_scal<Dtype>(prob_.count(), scale, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(OrdinalRegressionLossLayer)
#endif  // CPU_ONLY

INSTANTIATE_CLASS(OrdinalRegressionLossLayer);
REGISTER_LAYER_CLASS(OrdinalRegressionLoss);

}  // namespace caffe
