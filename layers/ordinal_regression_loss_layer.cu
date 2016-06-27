#include <cfloat>
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/ordinal_regression_loss_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void kernel_ordreg_forward(const int k, const int n,
    const Dtype* x, Dtype* y, const Dtype* label, const Dtype* weight, Dtype* loss) {
  CUDA_KERNEL_LOOP(idx, n) {
    const int sample_idx = idx / k;
    const int label_idx = idx % k;
    const int offset = 2*idx;
    const int this_label = static_cast<int>(label[sample_idx]);
    const Dtype this_weight = weight[label_idx];
    const Dtype* x_data = x + offset;
    Dtype* y_data = y + offset;
    Dtype* loss_data = loss + offset;
    Dtype max_input = max(x_data[0], x_data[1]);
    y_data[0] = exp(x_data[0] - max_input);
    y_data[1] = exp(x_data[1] - max_input);
    Dtype sum = y_data[0] + y_data[1];
    y_data[0] /= sum;
    y_data[1] /= sum;
    if (label_idx < this_label) {
      loss_data[0] = 0;
      loss_data[1] = -log(max(y[1], Dtype(FLT_MIN)));
    }
    else {
      loss_data[0] = -log(max(y[0], Dtype(FLT_MIN)));
      loss_data[1] = 0;
    }
    loss_data[0] *= this_weight;
    loss_data[1] *= this_weight;
  }
}

template<typename Dtype>
void OrdinalRegressionLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int n = bottom[0]->shape(0);
  const int nthread = n * k_;
  const Dtype* x = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* weight = weight_.gpu_data();
  Dtype* y = prob_.mutable_gpu_data();
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();  // reuse
  kernel_ordreg_forward<Dtype><<<CAFFE_GET_BLOCKS(nthread),
      CAFFE_CUDA_NUM_THREADS>>>(k_, nthread, x, y, label, weight, loss_data);
  Dtype loss;
  caffe_gpu_asum(bottom[0]->count(), loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / n;
}

template<typename Dtype>
__global__ void kernel_ordreg_backward(const int k, const int n,
    Dtype* dx, const Dtype* label, const Dtype* weight) {
  CUDA_KERNEL_LOOP(idx, n) {
    const int sample_idx = idx / k;
    const int label_idx = idx % k;
    const int offset = 2*idx;
    const int this_label = static_cast<int>(label[sample_idx]);
    const Dtype this_weight = weight[label_idx];
    Dtype* dx_data = dx + offset;
    if (label_idx < this_label) {
      dx_data[1] -= 1;
    }
    else {
      dx_data[0] -= 1;
    }
    dx_data[0] *= this_weight;
    dx_data[1] *= this_weight;
  }
}

template<typename Dtype>
void OrdinalRegressionLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int n = bottom[0]->shape(0);
    const int nthread = n * k_;
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* weight_data = weight_.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    kernel_ordreg_backward<Dtype><<<CAFFE_GET_BLOCKS(nthread),
        CAFFE_CUDA_NUM_THREADS>>>(k_, nthread, bottom_diff, label_data, weight_data);
    const Dtype scale = 1.0 / n;
    caffe_gpu_scal<Dtype>(prob_.count(), scale, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OrdinalRegressionLossLayer);

}  // namespace caffe
