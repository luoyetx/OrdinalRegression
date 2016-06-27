#include <cmath>
#include <vector>
#include <algorithm>
#include <boost/scoped_ptr.hpp>
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/ordinal_regression_loss_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template<typename TypeParam>
class OrdinalRegressionLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OrdinalRegressionLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(100, 200, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(100, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); i++) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 100;
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~OrdinalRegressionLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OrdinalRegressionLossLayerTest, TestDtypesAndDevices);

// TYPED_TEST(OrdinalRegressionLossLayerTest, TestGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   OrdinalRegressionLossLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_, 0);
// }

TYPED_TEST(OrdinalRegressionLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  scoped_ptr<OrdinalRegressionLossLayer<Dtype> > layer(
      new OrdinalRegressionLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype loss = this->blob_top_loss_->cpu_data()[0];
}

TYPED_TEST(OrdinalRegressionLossLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  scoped_ptr<OrdinalRegressionLossLayer<Dtype> > layer(
      new OrdinalRegressionLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  propagate_down.push_back(false);
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
}

}  // namespace caffe
