# Semantic Segmentation


[image1]: ./assets/encoder-decoder.jpg
[image2]: ./assets/summaries.png

# Project Implementation

## Network Architecture

Network is based on `FCN-8s` [[Shelhamer et al, 2016](https://arxiv.org/pdf/1605.06211.pdf)] with the following modifications:

- added scaling ops before applying 1x1 projections
- adam optimizer
- frozen VGG-16 encoder

It is an hourglass encoder-decoder architetcure with skip - connections.

![encoder-decoder][image1]

## Training Strategy
We trained for:
 - 30 epochs
 - inital learning rate 1e-3
 - batch size 16
 - weight decay after 250 iterations (approx 13 epochs)
 - step decay of 0.1
 - Adam Optimizer
 - dropout 0.5
 - L2 regularization for segmentation heads with weight 0.003

## Training evolution

You can see training evolution in summaries:

![summaries][image2]

Loss starts at 1.0 and rapidly decreases towards 0.3, and then slowly towards 0.14.



 
 
 
 
 

 
 


## Data Preprocessing

Added flip left right augmentation as tensor operations:

```python

def add_preprocessing(image_input, label_input, is_training):
  def _maybe_flip(input_tensor, mirror_cond, scope):
    return tf.cond(
      mirror_cond,
      lambda: tf.image.flip_left_right(input_tensor),
      lambda: input_tensor,
      name=scope)

  def _preprocess_train(image, label):
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)

    image = _maybe_flip(image, mirror_cond, scope="random_flip_image")
    label = _maybe_flip(label, mirror_cond, scope="random_flip_label")

    return image, label

  def _preprocess_test(image, label):
    return image, label

  def _map(fn, arrays, dtypes):
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtypes)
    return out

  mapper = \
    lambda img, label: tf.cond(
      tf.equal(
        is_training,
        tf.constant(True)),
      lambda: _preprocess_train(img, label),
      lambda: _preprocess_test(img, label))

  image_input, label_input = _map(mapper, [image_input, label_input],
                                  dtypes=(image_input.dtype, label_input.dtype))

  return image_input, label_input

```

 
## Implementation Details 

Added `Config` and TrainConfig with following options: 

```python

class TrainConfig(Config):
  EPOCHS = 30
  LEARNING_RATE = 1e-3
  BATCH_SIZE = 16
  LR_DECAY = 0.1
  DECAY_STEPS = 250


```




### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
