# CarND-Traffic-Sign-Classifier-Project
The goal in this project is to train a convolutional neural network with Tensorflow for building a classifier for the [German Traffic Sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Dataset consists of RGB images of size 32x32 for 43 different traffic signs. The aim is to get at least 93% accuracy on validation set.


[image1]: writeup_images/1.png
[image2]: writeup_images/2.png
[image3]: writeup_images/3.png
[image4]: writeup_images/4.png
[image5]: writeup_images/5.png
[image6]: writeup_images/6.png
[image7]: writeup_images/7.png
[image8]: writeup_images/8.png
[image9]: writeup_images/9.png
[image10]: writeup_images/10.png
[image11]: writeup_images/11.png
[image12]: writeup_images/12.png
[image13]: writeup_images/13.png
[image14]: writeup_images/14.png
[image15]: writeup_images/15.png
[image16]: writeup_images/16.png
[image17]: writeup_images/17.png
[image18]: writeup_images/18.png
[image19]: writeup_images/19.png
[image20]: writeup_images/20.png
[image21]: writeup_images/21.png
[image22]: writeup_images/22.png
[image23]: writeup_images/23.png
[image24]: writeup_images/24.png
[image25]: writeup_images/25.png
[image26]: writeup_images/26.png
[image27]: writeup_images/27.png
[image28]: writeup_images/28.png
[image29]: writeup_images/29.png
[image30]: writeup_images/30.png
[image31]: writeup_images/31.png
[image32]: writeup_images/32.png
[image33]: writeup_images/33.png
[image34]: writeup_images/34.png
[image35]: writeup_images/35.png
[image36]: writeup_images/36.png
[image37]: writeup_images/37.png
[image38]: writeup_images/38.png
[image39]: writeup_images/39.png
[image40]: writeup_images/40.png
[image41]: writeup_images/41.png
[image42]: writeup_images/42.png
[image43]: writeup_images/43.png
[image44]: writeup_images/44.png
[image45]: writeup_images/45.png
[image46]: writeup_images/46.png
[image47]: writeup_images/47.png
[image48]: writeup_images/48.png
[image49]: writeup_images/49.png
[image50]: writeup_images/50.png
[image51]: writeup_images/51.png
[image52]: writeup_images/52.png
[image53]: writeup_images/53.png
[image54]: writeup_images/54.png
[image55]: writeup_images/55.png
[image56]: writeup_images/56.png
[image57]: writeup_images/57.png
[image58]: writeup_images/58.png
[image59]: writeup_images/59.png
[image60]: writeup_images/60.png
[image61]: writeup_images/61.png
[image62]: writeup_images/62.png
[image63]: writeup_images/63.png
[image64]: writeup_images/64.png
[image65]: writeup_images/65.png
[image66]: writeup_images/66.png
[image67]: writeup_images/67.png
[image68]: writeup_images/68.png
[image69]: writeup_images/69.png
[image70]: writeup_images/70.png
[image71]: writeup_images/71.png
[image72]: writeup_images/72.png
[image73]: writeup_images/73.png
[image74]: writeup_images/74.png
[image75]: writeup_images/75.png
[image76]: writeup_images/76.png
[image77]: writeup_images/77.png
[image78]: writeup_images/78.png
[image79]: writeup_images/79.png
[image80]: writeup_images/80.png
[image81]: writeup_images/81.png
[image82]: writeup_images/82.png
[image83]: writeup_images/83.png
[image84]: writeup_images/84.png
[image85]: writeup_images/85.png
[image86]: writeup_images/86.png
[image87]: writeup_images/87.png
[image88]: writeup_images/88.png
[image89]: writeup_images/89.png
[image90]: writeup_images/90.png
[image91]: writeup_images/91.png
[image92]: writeup_images/92.png
[table]: writeup_images/table.png
[formula1]: writeup_images/formula1.png
[formula2]: writeup_images/formula2.png
[formula3]: writeup_images/formula3.png
[formula4]: writeup_images/formula4.png

---
**Exploration**

German Traffic Sign dataset consists of three separate parts, namely training (34799 examples), validation (4410 examples) and test sets (12630 examples). Images are all in RGB and resized to the size of 32x32. In total, there are 43 distinct traffic signs.
Here are 10 examples from each type:
![image1]
![image2]
![image3]
![image4]
![image5]
![image6]
![image7]
![image8]
![image9]
![image10]
![image11]
![image12]
![image13]
![image14]
![image15]
![image16]
![image17]
![image18]
![image19]
![image20]
![image21]
![image22]
![image23]
![image24]
![image25]
![image26]
![image27]
![image28]
![image29]
![image30]
![image31]
![image32]
![image33]
![image34]
![image35]
![image36]
![image37]
![image38]
![image39]
![image40]
![image41]
![image42]
![image43]
Some of the images are dark, whereas the others are very light, 
 not mentioning the ones that are hazy and difficult also for human eye 
  to label them correctly. 
  
  The dataset is imbalanced and the class distribution in different sets are comparable:
  ![table]
  
  Some of the darkest images in training data:
  ![image44]
  
 **Preprocessing**
 
Since the traffic signs are distinct by their appearances, we can omit the color information and convert the images to 
the grayscale. Conversion to grayscale makes dark images more distinguishable and also improved the classification results for me. Here is how the dark image set above
looks like in grayscale:
![image45]


To improve the contrast of the images I utilize two methods:
 * [Histogram equalization]([adaptive histogram equalization](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html))
 * [Local contrast normalization]((http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf))

 
 For local contrast normalization, first, (normalized) gaussian weighted sum of surrounding pixel intensities are subtracted from pixels.
  
  ![formula1]
 
 At second step, these values are divided by square root of weighted sum of squares of all features over a spatial neighborhood.
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![formula2]

 where 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![formula3] 
 
 and 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![formula4]
  
 As a result, dark and light regions change a little, but the edges of the images become clearer.
 ![image46]
 Another method to make contrast in images more visible is to use contrast limited adaptive histogram equalization, which spreads the intensities to a wider range, thus improving the contrast.
 ![image47]

**Augmentations**

I artificially increase the size of dataset by adding distorted images to the original data, in order to make 
the classifier more robust to potential deformations. Augmentations include rotations by angle uniformly selected 
from interval of [-20, 20] degrees and uniform distortions by magnitude of [-0.1, 0.1] of the image applied to each corner.
For each image in the training dataset I create an additional randomly rotated image. Here are some examples:
 
 ![image48]
 
Then for each of the images in this new dataset I add one image which is distorted at each of the four edges randomly.
 
 ![image49]
 
As a result training data grows to 4x of the original size.
 
 **Architecture**
 
  I used a neural network architecture similar to [Cireşan, 2012](http://people.idsia.ch/~juergen/nn2012traffic.pdf).  
  
  Layer|
  ------------- |
  convolutional, 100 features, 7x7 kernel      |
  dropout, probability 0.3 |
  maxpool, 2x2 kernel, 2x2 strides|
  convolutional, 150 features, 4x4 kernel      |
  dropout, probability 0.3 |
  maxpool, 2x2 kernel, 2x2 strides|
  convolutional, 250 features, 2x2 kernel      |
  dropout, probability 0.3 |
  maxpool, 2x2 strides|
  dense, size 300|
  dropout, probability 0.35 |
  softmax, 43|
  
For training the model I use [Adam](https://arxiv.org/pdf/1412.6980.pdf) (adaptive momentum) optimizer 
with the batch size of 64 and train the model maximum for 50 epochs. Early stopping acts as a regularizer 
and lets us stop training before it starts overfitting to the training data. I use the value of the loss function on 
validation data to decide when to stop. Dropout probabilities that I run the training with are 0.3 after convolutional layers 
and 0.35 for the dense layer. While training, I selected the combination of simplicity of the model and the speed for training as priority.
 However, one can elaborate on this task as described [here](https://arxiv.org/pdf/1511.02992.pdf).

**Solution**

Since I use fairly simple architecture, I let the model run 5 times. I repeat the same procedure for both preprocessing methods (contrast normalization
 and histogram equalization), which results in 10 models. I average the output 
probabilities of each model to make the final prediction. Accuracy reached on train and validation sets reaches 99% for each of the models. The accuracy 
on test set ranges between 97.5-98.5% for the models and the average of the probabilities gives ~99% accuracy on test data.

 precision    recall  f1-score   support

          0       1.00      1.00      1.00        60
          1       0.99      1.00      1.00       720
          2       1.00      1.00      1.00       750
          3       0.99      0.97      0.98       450
          4       1.00      1.00      1.00       660
          5       0.98      0.99      0.99       630
          6       1.00      0.99      0.99       150
          7       0.99      1.00      1.00       450
          8       1.00      1.00      1.00       450
          9       0.99      1.00      0.99       480
         10       1.00      1.00      1.00       660
         11       0.93      0.98      0.95       420
         12       0.99      1.00      1.00       690
         13       1.00      1.00      1.00       720
         14       0.99      1.00      1.00       270
         15       1.00      1.00      1.00       210
         16       1.00      1.00      1.00       150
         17       1.00      0.99      0.99       360
         18       0.99      0.97      0.98       390
         19       1.00      1.00      1.00        60
         20       0.97      1.00      0.98        90
         21       0.90      1.00      0.95        90
         22       1.00      0.84      0.91       120
         23       0.94      1.00      0.97       150
         24       1.00      0.99      0.99        90
         25       0.95      0.99      0.97       480
         26       0.99      1.00      0.99       180
         27       0.97      0.50      0.66        60
         28       0.99      1.00      1.00       150
         29       0.96      1.00      0.98        90
         30       0.93      0.87      0.90       150
         31       1.00      1.00      1.00       270
         32       1.00      1.00      1.00        60
         33       1.00      1.00      1.00       210
         34       1.00      1.00      1.00       120
         35       1.00      0.99      0.99       390
         36       0.99      0.98      0.99       120
         37       1.00      1.00      1.00        60
         38       1.00      1.00      1.00       690
         39       1.00      1.00      1.00        90
         40       0.99      0.97      0.98        90
         41       1.00      0.90      0.95        60
         42       1.00      1.00      1.00        90

avg / total       0.99      0.99      0.99     12630


**Test a Model on New Images**

To check the classification results on new images, I found 10 images (three of them I shot myself) that seemed interesting to me. 
Only three of the images (dirty and yellowish "No entry" signs and "Priority road") have the type in German Traffic Sign Dataset. Rest 
of the images have something in common in appearance with the signs in our dataset and the expectation, that the classes, which are 
predicted with high probability for those images, correspond to similarly looking signs, holds for almost in all cases.




 ![image50]
 ![image51]
 ![image52]
 ![image53]
 
 
![image54]
![image55]
![image56]
![image57]
  
![image58]
![image59]
![image60]
![image61]


![image62]
![image63]
![image64]
![image65]

![image66]
![image67]
![image68]
![image69]

![image70]
![image71]
![image72]
![image73]


![image74]
![image75]
![image76]
![image77]


![image78]
![image79]
![image80]
![image81]

![image82]
![image83]
![image84]
![image85]


![image86]
![image87]
![image88]
![image89]

![image90]
![image91]
![image92]