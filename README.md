# FORMULA
## A Deep Learning Approach for Rare Alarms Predictions in Industrial Equipment
We present a novel deep learning-based approach called FORMULA with a new formulation for the Alarm Forecasting problem.


### Abstract
Predictive Maintenance technologies are particularly appealing for Industrial
Equipment producers, as they pave the way to the selling of high added-value services
and customized maintenance plans. However, standard Predictive Maintenance
approaches assume the availability of sensor measurements, and the costs associated
with adding sensors or remotely accessing sensor readings may discourage the
development of such technologies. In this context, Alarm Forecasting can be very
useful as it represents a low-cost alternative or helpful support to sensor-based
Predictive Maintenance. In this work, we propose a new formulation for the Alarm
Forecasting problem, framed as a multi-label classification task. We present a novel
deep learning-based approach inspired by Natural Language Processing, called
FORMULA (alarm FORecasting in MUlti-LAbel setting).

To cope with alarm imbalance, we draw inspiration from Segmentation and Object
Detection, and we consider a suitable loss function, namely Weighted Focal Loss,
which turns out to be very effective in predicting rare alarms. These alarms, even if
they are difficult to predict by nature, often are business-critical.

We assess the proposed approach on a representative real-world problem from the
packaging industry. In particular, we show that it outperforms not only classic multilabel
techniques but also models based on recurrent neural networks. As regards the latter,
the proposed approach also exhibits a lower computational burden, both in terms of
training time and model size. To foster research in the field and reproducibility we also
publicly share the alarm logs dataset and the code used to perform the experiments.