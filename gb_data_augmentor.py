from auto_augmenter_utils import Sequential_add_bbs_only
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def sub_policy_1():
    seq = Sequential_add_bbs_only([
        # translateX 0.6, 4, range[-150, 150]
        iaa.Sometimes(0.6,
                      iaa.Affine(translate_px={'x': -50, 'y': 0})
                      ),
        # equalize, 0.8, 10, range[None]
        iaa.Sometimes(0.8,
                      iaa.Affine(rotate=20))
    ], bbox_only=[0, 0])

    return seq


def sub_policy_2():
    seq = Sequential_add_bbs_only([
        # bbox_only_TranslateY, 0.2, 2, range[-150, 150]
        iaa.Sometimes(0.5, iaa.Affine(
            translate_px={'x': 0, 'y': -100},
        )),
        # Cutout, 0.8, 8, range[0, 60]
        iaa.Sometimes(0.8, iaa.CoarseDropout(p=0.1, size_px=40))
    ], bbox_only=[1, 0])

    return seq


def sub_policy_3():
    seq = Sequential_add_bbs_only([
        # ShearY 1.0, 2, range[-0.3, 0.3]
        iaa.Sometimes(1.0, iaa.Affine(
            shear=(-35, 35) # degree
        )),
        # Bbox_only_translateY 0.6, 6, range[-150, 150]
        iaa.Sometimes(0.6, iaa.Affine(
            translate_px={'x': 0, 'y': 0}
        ))
    ], bbox_only=[1, 0])

    return seq


def sub_policy_4():
    seq = Sequential_add_bbs_only([
        # Rotate 0.6 10, range[-30, 30]
        iaa.Sometimes(0.6, iaa.Affine(
            rotate=20,
        )),
        # Color 1.0, 6, range[0.1, 1.9], 1.3
        # if the factor inside
        # iaa.Sometimes(1, iaa.Alpha(1.3, iaa.Grayscale(1.0)))
        iaa.Sometimes(1, iaa.MultiplyElementwise(1.3))
    ], bbox_only=[0, 0])

    return seq


def sub_policy_5():
    seq = Sequential_add_bbs_only([
        # 0 probability, do nothing
        iaa.Sometimes(0, iaa.Affine(
            translate_px={'x': -50, 'y': 0}
        )),
        # 0 probability, do nothing
        iaa.Sometimes(0, iaa.AllChannelsHistogramEqualization())
    ], bbox_only=[0, 0])
    return seq


def apply(imgs, batch_bbs):
	# constract the BoundingBoxesOnImage data
    batch_bbs_iaa = []
    for i, img in enumerate(imgs):
        one_img_bbs_list = []
        for bbox in batch_bbs[i]:
            bbox_11 = BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
            one_img_bbs_list.append(bbox_11)
        bbox_in1 = BoundingBoxesOnImage(one_img_bbs_list, shape=img.shape)
        batch_bbs_iaa.append(bbox_in1)
	
	# select sub_policy_3
    seq = sub_policy_3()
    
	# do image aug
	images_aug, bbs_aug = seq(images=imgs, bounding_boxes=batch_bbs_iaa)
	# optimize some coord out of the image.
	for j in range(len(bbs_aug)):
		bbs_aug[j] = bbs_aug[j].remove_out_of_image().clip_out_of_image()
	
	# return batch of image, batch of bbs after aug. 
    return images_aug, bbs_aug

if __name__=="__main__":
    import cv2
    img = cv2.imread('./data_loader/d4236121.jpg')
    bbs = [[187, 93, 511, 341], [0, 0, 88, 50]]

    im_aug, bbs_aug = apply([img], batch_bbs=[bbs])