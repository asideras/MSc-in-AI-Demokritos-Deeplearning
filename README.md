This repository includes the code and report for the course project on Deep Learning in the MSc in AI program offered by NCSR Demokritos and the University of Piraeus. The full report can be found [here](Classification___Localization_of_Inpainted_Regions.pdf).

Our objective was to train a model capable of detecting artificial regions, which are regions generated by AI. Image inpainting is a computer vision technique, where an algorithm attempts to reconstruct or fill in a masked region using information from the surrounding area. To accomplish this, we created a dataset (based on [Places365](http://places2.csail.mit.edu/download.html)) by randomly applying masks to images ([original](Data%20Samples/original/) & [original_masked](Data%20Samples/original_masked/)). Subsequently, we utilized a GAN network trained specifically for this task to fill these regions. We use the DeepFillv2_Pytorch, basically a Pytorch re-implementation for the paper [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589), from this [repo](https://github.com/csqiangwen/DeepFillv2_Pytorch#readme). <br>

Now that we have a dataset comprising of images that contain an artificial region, we need to train our own model that detects such regions (one per image). Below is an example that illustrates all the steps made for a single image. 

![Full process](Data%20Samples/full.png)

## Contents

- [Data samples](Data%20Samples): A folder that contains some samples of the images for illustration purposes.
   - [inpainted](Data%20Samples/inpainted/) folder contains the final edited images that our model gets as input.
   - [masks](Data%20Samples/masks/) folder contains the corresponding mask for each masked image that DeepFillv2_Pytorch gets as input.
   - [original](Data%20Samples/original/) folder contains the original images from Places365.
   - [original_masked](Data%20Samples/original_masked/) folder contains the images that DeepFillv2_Pytorch got as input (along with the corresponding masks).
- [Model](Model): A folder that contains the python files that defines our neural network models
   - [Losses.py](Losses.py) contains some implemented loss functions we used.
   - [data_loader.py](data_loader.py) is a script that loads the data from the correct folder (original or inpainted, as our model has to distinguish between them), splits them into batches and feed them to the model.
  - [network.py](network.py) loads pretrained models and creates the corresponding classes in order to manupulate them properly.
- [Utils](Utils): A folder that contains some helper scripts.
   - [IoC.py](IoC.py): A script that contains the computation of Intersection Over Union metric used as performance metric for the localization part.
   - [create_masks.py](create_masks.py): A script used for the mask creation. It produces a uniformly mask box and saves this mask to a different image.
   - [draw_boxes.py](draw_boxes.py): Contains method that actually draws the predicted bounding box. And the method demonstrate_result, that loads the final model and make inferences about images
- [anns_class_local.csv](anns_class_local.cs) contains the ground truth labels for each image. It's a 5-dimensional vector with a boolean value representing whether the image has been inpainted or not, and the 4 coordinates (xmin,ymin,xmax,ymax) that locate the inpainted region.
- [demo.ipynb](demo.ipynb) is a jupyter notebook that illustrates some random sampled results. It imports and uses the demonstrate_result method from Utils.draw_boxes.
- [requirements.txt](requirements.txt) is a file that contains all the packages needed to run the code.
- [test.py](test.py) contains the testing code.
- [train.py](train.py) contains all the training code.
