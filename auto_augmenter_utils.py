from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class Sequential_add_bbs_only(iaa.Sequential):
    '''
    Inherit from the iaa.Sequential.
    Add bbs only ops than the original one.
    '''
    def __init__(self, children=None, random_order=False, name=None, deterministic=False, random_state=None, bbox_only=None):
        super().__init__(children=children, random_order=random_order, name=name, deterministic=deterministic, random_state=random_state)
        # list of boolean vals, [1, 0, 0, 1], 1-bbox only, 0-the whole image
        self.bbs_only = bbox_only


    def __call__(self, *args, **kwargs):
        '''
        Augment the imgs, bbs may be [[img1bb1, img1bb2], [img2bb1, img2bb2], [...]]
        :param args:
        :param kwargs:
            images: iterable of images, [img1, img2, img3, ...]
            bounding_boxes: iterable of BBs, [bbs1, bbs2, bbs3,...], type=BoundingBoxesOnImage
            ** Besides, the bounding_boxes.__len__() should equal to the images.__len__()
        :return: aug_imgs, aug_bbs
        '''
        self.ims = kwargs.get('images', None)
        self.batch_bbs = kwargs.get('bounding_boxes', None)

        assert len(self.ims) == len(self.batch_bbs), "The len of imgs and len of bbs should be the same !"
        # augmenters are in the self
        self.augmenters = self[:]

        # iterate through the self.augmenters, (bbox_only ops) if ele==1 else (the whole image ops)
        for i, ele in enumerate(self.bbs_only):
            if ele:
                # loop through all the images
                for j, img in enumerate(self.ims):
                    # loop through all the bbs in one image
                    for k, bbox in enumerate(self.batch_bbs[j].bounding_boxes):
                        # get the bbox patch
                        bb_patch = img[bbox.y1_int:bbox.y2_int, bbox.x1_int:bbox.x2_int, :]
                        bb_coord = BoundingBoxesOnImage(
                            [BoundingBox(x1=0, y1=0, x2=bb_patch.shape[1], y2=bb_patch.shape[0])],
                        shape=bb_patch.shape)
                        # do augment on the bbox patch
                        aug_bb_patch, aug_bb_coord = self.augmenters[i](image=bb_patch, bounding_boxes=bb_coord)

                        # repad to orig images
                        self.ims[j][bbox.y1_int:bbox.y2_int, bbox.x1_int:bbox.x2_int, :] = aug_bb_patch
                        # optimize the aug_bb_coord in case the coord out of the bbox patch size
                        aug_bb_coord = aug_bb_coord.remove_out_of_image().clip_out_of_image()
                        # add relative coord to the whole image coord, offset of left-top coord
                        self.batch_bbs[j].bounding_boxes[k] = self.add_bbs(self.batch_bbs[j].bounding_boxes[k],
                                                                      aug_bb_coord.bounding_boxes[0])
            else:
                # no need to extract the bbox patch, do augment on iterables of images and bounding boxes
                self.ims, self.batch_bbs = self.augmenters[i](images=self.ims, bounding_boxes=self.batch_bbs)
                # deal with the bbox coord out of the images.
                for i in range(len(self.batch_bbs)):
                    self.batch_bbs[i] = self.batch_bbs[i].remove_out_of_image().clip_out_of_image()

        return self.ims, self.batch_bbs

    @classmethod
    def add_bbs(self, bb1, bb2):
        x1 = bb1.x1 + bb2.x1
        y1 = bb1.y1 + bb2.y1
        x2 = bb1.x1 + bb2.x2
        y2 = bb1.y1 + bb2.y2

        return BoundingBox(x1=x1, y1=y1, x2=x2,y2=y2)