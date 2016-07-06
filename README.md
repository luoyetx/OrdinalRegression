Ordinal Regression
==================

Caffe Loss Layer for `Ordinal Regression with Multiple Output CNN for Age Estimation`.

## How to

You need to install [Caffe][caffe] first. Copy relative files to Caffe's source code tree.

```
export CAFFE_HOME=/path/to/caffe
cp layers/ordinal_regression_loss_layer.hpp $CAFFE_HOME/include/caffe/layers/ordinal_regression_loss_layer.hpp
cp layers/ordinal_regression_loss_layer.cpp $CAFFE_HOME/src/caffe/layers/ordinal_regression_loss_layer.cpp
cp layers/ordinal_regression_loss_layer.cu $CAFFE_HOME/src/caffe/layers/ordinal_regression_loss_layer.cu
cp layers/test_ordinal_regression_loss_layer.cpp $CAFFE_HOME/src/caffe/test/test_ordinal_regression_loss_layer.cpp
```

Modify `$CAFFE_HOME/src/caffe/proto/caffe.proto` according to `layers/caffe.proto`

After all, follow Caffe's documents and compile it.

## Test the layer

`make runtest GTEST_FILTER='OrdinalRegressionLossLayerTest/*'`

## References

- [Caffe][caffe]
- [Ordinal Regression with Multiple Output CNN for Age Estimation](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Niu_Ordinal_Regression_With_CVPR_2016_paper.pdf)


[caffe]: https://github.com/BVLC/caffe
