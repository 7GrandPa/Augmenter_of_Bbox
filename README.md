# Augmenter_of_Bbox
reference: Learning Data Augmentation Strategies for Object Detection

Augment bbox-only patch or the whole images using imgaug module

# start
See gb_data_augmentor.py, 
1. Import the auto_augmenter_utils;
2. Sequential_add_bbs_only(...., bbox_only=[0,1,0,1], 1 means corresponding augmenter operate bbox_only.
3. augment the images, and finally you'll get the images and bboxes after aug.

## Shear
  ![image](https://github.com/7GrandPa/Augmenter_of_Bbox/blob/master/out_data/1.jpg)
## Colorspace and translate_x
  ![image](https://github.com/7GrandPa/Augmenter_of_Bbox/blob/master/out_data/2.jpg)
