# Training a TensorFlow.js model for Speech Commands Using Browser FFT

<style>
 @import url('//maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css');
 
.isa_info, .isa_success, .isa_warning, .isa_error {
margin: 10px 0px;
padding:12px;
 
}
.isa_info {
    color: #00529B;
    background-color: #BDE5F8;
}
.isa_success {
    color: #4F8A10;
    background-color: #DFF2BF;
}
.isa_warning {
    color: #9F6000;
    background-color: #FEEFB3;
}
.isa_error {
    color: #D8000C;
    background-color: #FFD2D2;
}
.isa_info i, .isa_success i, .isa_warning i, .isa_error i {
    margin:10px 22px;
    font-size:2em;
    vertical-align:middle;
}
</style>

! 
ref from https://github.com/tensorflow/tfjs-models/tree/master/speech-commands/training/browser-fft
it is not my code, howerver i add some comment to make u easily understand it.
###


This directory contains two example notebooks. They demonstrate how to train
custom TensorFlow.js audio models and deploy them for inference. The models
trained this way expect inputs to be spectrograms in a format consistent with
[WebAudio's `getFloatFrequencyData`](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode/getFloatFrequencyData).
Therefore they can be deployed to the browser using the speech-commands library
for inference.

Specifically,

- [training_custom_audio_model_in_python.ipynb](./training_custom_audio_model_in_python.ipynb)
  contains steps to preprocess a directory with audio examples stored as .wav
  files and the steps in which a tf.keras model can be trained on the
  preprocessed data. It then demonstrates how the trained tf.keras model can be
  converted to a TensorFlow.js `LayersModel` that can be loaded with the
  speech-command library's `create()` API. In addition, the notebook also shows
  the steps to convert the trained tf.keras model to a TFLite model for
  inference on mobile devices.
- [tflite_conversion.ipynb](./tflite_conversion.ipynb) illustrates how
  an audio model trained on [Teachable Machine](https://teachablemachine.withgoogle.com/train/audio)
  can be converted to TFLite directly.
