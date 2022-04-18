## Implementation of Neural Style Transfer;

* The implementation is based on [Gatys et al., 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf);

### Example style transfer:

* The style image is the starry night. The content image is a photo of houses in Copenhagen I took while on holidays.


![alt text](https://github.com/mariovas3/comp_vision_repro/blob/master/style_transfer_repro/data/images/houses_denmark_540p_350iter.jpg)


### Some info about my implementation:
* I used the same definition for the style loss as in [Gatys et al., 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).
I used the Gram matrices from the representations of the first conv module from each VGG block. As in the paper I assign
an equal weight (one over number of style layers) to the style loss from each layer.
* I also chose the second conv module from the second VGG block to obtain representations
for the content image. This seems to have enforced a stronger restriction on modifying the content of the image (as desired) compared
to the second conv module of the fourth VGG block (as described in the paper).
* Instead of sum of squares loss for the content loss, I used mean squeared error loss. Seemed to work better in my application.
* Finally, I used the L-BFGS algorithm to perform unconstrained optimisation on the synthetic image. Initially I tried out ADAM with
a learning rate schedule, however, this didn't seem to train the model as fast as L-BFGS did. In the paper the authors also note they
use L-BFGS.
* To sum up, the biggest trick here is to choose a good trade-off between the weights for the style and content losses.
