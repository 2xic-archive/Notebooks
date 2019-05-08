
# Computer vision and machine learning notebooks
Some algorithms I have implemented recently. Most of the vision notebooks are based off ideas from the book "Computer Vision - Algorithms and Applications" by Springer. The machine learning models are a bit of everything. Some basic, some more advanced stuff. You need to show a spectrum of knowledge.

**Locally adaptive histogram**

<img src="README/locally adaptive histogram.jpg" width="500px" />

**Qlearning**

I wrote a basic game to test q-learning.

<img src="README/qlearning.gif" height="250px" />

**Anti-aliasing**

<img src="README/anti-aliasing.jpg" width="500px"/>

**Gan**

based off https://arxiv.org/abs/1610.09585

<img src="README/gan.gif" height="250px" />

**Dropout**

based off http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
model without dropout

<img src="README/not_dropout.jpg" height="250px" />

model with dropout

<img src="README/dropout.jpg" height="250px" />

**Feature checker**

Based off the idea from https://github.com/evilsocket/ergo (relevance.py).
What feature is contributing what to the models results? If you are going to use this 
in production, set some random data points to zero during trainings as well.

<img src="README/model_check.jpg" height="250px" />

**Filters fixing noise**

<img src="README/filtes_image.jpg" width="500px" />

**Gradient descent**

Finding a local minima

**A\***

Search algorithm.

**K-means clustering**

<img src="README/k-means clustering.jpg" height="250px" />

**LeCun CNN**

Implemented LeChun CNN model, based off http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

**Markov Random Field**

<img src="README/MRF.jpg" width="500px" />

**Hough Transform**

Basic implementation of hough Transform.

**Auto encoder**

<img src="README/auto_encoder.jpg"  width="500px" />

**Gan in numpy**

<img src="README/np_gan.jpg"  height="250px" />

**Segmentation graph**

<img src="README/segmentation_grapt.jpg"  height="250px" />

**MNIST generalization test**

got the idea from https://arxiv.org/pdf/1611.03530.pdf , how much noise can a simple model handle and still do good evaluation on a noise free dataset? 

<img src="README/generalization_test_bar.jpg"  height="250px" />

the accuracy over time is based on the traing data.

<img src="README/generalization_test.jpg"  height="250px" />

**Generalization**

https://arxiv.org/pdf/1611.03530.pdf

<img src="README/generalization_paper_acc.jpg"  height="250px" />

<img src="README/generalization_paper_loss.jpg"  height="250px" />

**Poisson image editing**

Mostly a fork off [this](https://github.com/willemmanuel/poisson-image-editing) implementation, I fixed support for python3 and made it work with all the channels. Removed opencv depency as well.

<img src="README/poisson_image_editing.jpg"  width="500px" />

**Feature detection**

Harris corner detector to find special features in each image. Using MSE to connect the special features.

<img src="README/feature_detection.jpg"  width="500px" />


**Counterfactual regret minimization**

Python implementation of the rock, paper, scissor section of http://modelai.gettysburg.edu/2013/cfr/cfr.pdf 

**Lucas–Kanade method, optical flow**

<img src="README/optical_input.jpg"  height="250px" />
<img src="README/optical1.jpg"  height="250px" />
<img src="README/optical2.jpg"  height="250px" />
<img src="README/optical3.jpg"  height="250px" />
<img src="README/optical4.jpg"  height="250px" />


**Segnet**

Had memoryerrors so this model was trained on only a subset of the training set (please hire me so I can build a computer for machine learning). I used the CamVid dataset.

<img src="README/segnet.jpg"  width="500px" />


**Transfer learning**

https://en.wikipedia.org/wiki/Transfer_learning

**Autoencoder fixing image noise**

<img src="README/autoencoder_noise.jpg"  width="500px" />

**Numpy rnn**

Loosely based on [iamtrask](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/) post. Maybe I wouldn't have coded this if he had used a linked list.



