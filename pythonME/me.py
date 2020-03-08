import numpy
from pythonME import pyME

class ME:
    estimator = None
    width = None
    height = None
    pixel_precision = None

    def __init__(self, width, height, precision="QuarterPixel", min_size_block=4, max_size_block=16, loss_metric="sad",
                 max_len_hor=None, max_len_vert=None):
        convert_precision_to_int = {"OnePixel": 1, "HalfPixel": 2, "QuarterPixel": 4}
        block_sizes = [4, 8, 16]
        metrics = ["sad", "colorindependent"]
        assert isinstance(width, int), "Frame width must integer! width = {}".format(width)
        assert width % 16 == 0, "Frame width must be a multiply 16! width = {}".format(width)
        assert isinstance(height, int), "Frame height must integer! height = {}".format(height)
        assert height % 16 == 0, "Frame height must be a multiply 16! height = {}".format(height)
        assert precision in convert_precision_to_int.keys(), "Wrong precision: {}! Variants: {}".format(precision, list(
            convert_precision_to_int.keys()))
        assert min_size_block in block_sizes, "Wrong min_size_block: {}! Variants: {}".format(min_size_block,
                                                                                              block_sizes)
        assert max_size_block in block_sizes, "Wrong max_size_block: {}! Variants: {}".format(max_size_block,
                                                                                              block_sizes)
        assert loss_metric.lower() in metrics, "Wrong loss_metric: {}! Variants: {}".format(loss_metric.lower(),
                                                                                            metrics)
        if max_len_hor is None:
            max_len_hor = round(0.12 * width)
        if max_len_vert is None:
            max_len_vert = round(0.12 * height)
        assert isinstance(max_len_hor, int), "max_len_hor must integer! max_len_hor = {}".format(max_len_hor)
        assert isinstance(max_len_vert, int), "max_len_vert must integer! max_len_vert = {}".format(max_len_vert)
        self.pixel_precision = convert_precision_to_int[precision]
        self.width = width
        self.height = height
        self.estimator = pyME.ME()
        self.estimator.InitME(width, height, self.pixel_precision, min_size_block, max_size_block, loss_metric.lower(),
                              max_len_hor, max_len_vert)

    def EstimateME(self, current_frame, reference_frame, return_error=False):
        if not isinstance(current_frame, numpy.ndarray):
            current_frame = numpy.array(current_frame)
        current_frame = current_frame.astype(numpy.uint8)
        if not isinstance(reference_frame, numpy.ndarray):
            reference_frame = numpy.array(reference_frame)
        reference_frame = reference_frame.astype(numpy.uint8)
        assert current_frame.shape == (
        self.height, self.width, 3), "current_frame.shape must be equal ({}, {}, 3)! current_frame.shape = {}".format(
            self.height, self.width, current_frame.shape)
        assert reference_frame.shape == (self.height, self.width,
                                         3), "reference_frame.shape must be equal ({}, {}, 3)! reference_frame.shape = {}".format(
            self.height, self.width, reference_frame.shape)
        # check images sizes
        result = self.estimator.EstimateME(current_frame, reference_frame).copy()
        if return_error:
            return (numpy.transpose(result[:, :, 0] / self.pixel_precision, axes=[1, 0]).reshape(self.height, self.width),
                    numpy.transpose(result[:, :, 1] / self.pixel_precision, axes=[1, 0]).reshape(self.height, self.width),
                    numpy.transpose(result[:, :, 2], axes=[1, 0]).reshape(self.height, self.width))
        else:
            return (numpy.transpose(result[:, :, 0] / self.pixel_precision, axes=[1, 0]).reshape(self.height, self.width),
                    numpy.transpose(result[:, :, 1] / self.pixel_precision, axes=[1, 0]).reshape(self.height, self.width))

    def __del__(self):
        if not self.estimator is None:
            self.estimator.DeinitME()


def TestME():
    import os
    # from skimage.io import imsave
    prefix = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(prefix, "test", "l.png.my"), "rb") as f:
        data = f.read()
        l = [i for i in data]
    l = numpy.array(l).astype(numpy.uint8)[:3 * 1024 * 432].reshape((432, 1024, 3))

    with open(os.path.join(prefix, "test", "r.png.my"), "rb") as f:
        data = f.read()
        r = [i for i in data]
    r = numpy.array(r).astype(numpy.uint8)[:3 * 1024 * 432].reshape((432, 1024, 3))
    # imsave("r.bmp", r)
    # imsave("l.bmp", l)

    a = pyME.ME()
    a.InitME(1024, 432, 4, 4, 16, "sad", 500, 20)
    l_to_r_motion_filename = str(os.path.join(prefix, "test", "pyME_l_to_r.motion"))
    a.LogME(l, r, l_to_r_motion_filename)
    a.DeinitME()
    b = pyME.ME()
    b.InitME(1024, 432, 4, 4, 16, "sad", 500, 20)
    r_to_l_motion_filename = str(os.path.join(prefix, "test", "pyME_r_to_l.motion"))
    b.LogME(r, l, r_to_l_motion_filename)
    b.DeinitME()

    with open(os.path.join(prefix, "test", "GT_l_to_r.motion"), "r") as f_gt, open(l_to_r_motion_filename,
                                                                                   "r") as f_tested:
        data_gt = f_gt.read().split("\n")
        data_tested = f_tested.read().split("\n")
        assert len(data_gt) == len(data_tested), "l_to_r.motion: motion files has different lens"
        check = [True if gt != tested else False for gt, tested in zip(data_gt, data_tested)]
        if numpy.any(check):
            idx = check.index(True)
            raise AssertionError("l_to_r.motion: motion files different in line {}".format(idx + 1))

    with open(os.path.join(prefix, "test", "GT_r_to_l.motion"), "r") as f_gt, open(r_to_l_motion_filename,
                                                                                   "r") as f_tested:
        data_gt = f_gt.read().split("\n")
        data_tested = f_tested.read().split("\n")
        assert len(data_gt) == len(data_tested), "l_to_r.motion: motion files has different lens"
        check = [True if gt != tested else False for gt, tested in zip(data_gt, data_tested)]
        if numpy.any(check):
            idx = check.index(True)
            raise AssertionError("l_to_r.motion: motion files different in line {}".format(idx + 1))
    print('All tests passed')
