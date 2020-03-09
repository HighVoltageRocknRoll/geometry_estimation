import numpy as np
from pythonME.me import ME

class MEHandler(object):
    def __init__(self, H, W, loss_metric, runs_to_warm_up=2):
        self.W = W
        self.H = H
        self.loss_metric = loss_metric
        self.runs_to_warm_up = runs_to_warm_up
        self.L2R_ME = ME(W, H, loss_metric=loss_metric)
        self.R2L_ME = ME(W, H, loss_metric=loss_metric)

    def warmed_up_me(self, MEInstance, cur_img, ref_img):
        for _ in range(self.runs_to_warm_up - 1):
            MEInstance.EstimateME(cur_img, ref_img)
        return MEInstance.EstimateME(cur_img, ref_img)

    def calculate_disparity(self, img_l, img_r):
        # img_l, img_r - images (HxWx3) shape. H and W must be a multiple of 16. 
        # return - tuple of 2 tensors of (2, H//4, W//4) shape -- Motion Vectors from img_l to img_r and back. # div by 4 because of 4x4 min MB size
        l2r = np.asarray(self.warmed_up_me(self.L2R_ME, img_l, img_r))
        r2l = np.asarray(self.warmed_up_me(self.R2L_ME, img_r, img_l))

        return l2r[..., ::4, ::4], r2l[..., ::4, ::4]

### Code from https://stackoverflow.com/questions/34152758/how-to-deepcopy-when-pickling-is-not-possible
### Allow torch.Dataloader to pickle MEHandler (instanses of pyME)

import copyreg

def pickle_ME(me):
    return MEHandler, (me.H, me.W, me.loss_metric, me.runs_to_warm_up)

copyreg.pickle(MEHandler, pickle_ME)
#
###