from __future__ import print_function, division
import os
from os.path import exists
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.pf_dataset import PFDataset, PFPascalDataset
from data.caltech_dataset import CaltechDataset
from data.tss_dataset import TSSDataset
from data.dataset_3d import Dataset3D
from data.dataset_3d_me import Dataset3DME
from data.download_datasets import *
from geotnf.point_tnf import *
from geotnf.transformation import GeometricTnf
from image.normalization import NormalizeImageDict
from model.cnn_geometric_model import CNNGeometric
from options.options import ArgumentParser
from util.torch_util import BatchTensorToVars, str_to_bool
from util.eval_util import pck_metric, area_metrics, flow_metrics, compute_metric
from util.dataloader import default_collate

"""

Script to evaluate a trained model as presented in the CNNGeometric TPAMI paper
on the PF/PF-pascal/Caltech-101 and TSS datasets

"""

def main(passed_arguments=None):

    # Argument parsing
    args,arg_groups = ArgumentParser(mode='eval').parse(passed_arguments)
    print(args)

    # check provided models and deduce if single/two-stage model should be used
    two_stage = args.model_2 != ''
     
    use_cuda = torch.cuda.is_available()
    use_me = args.use_me

    print('Creating CNN model...')

    def create_model(model_filename):
        checkpoint = torch.load(model_filename, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        output_size = checkpoint['state_dict']['FeatureRegression.linear.bias'].size()[0]

        if output_size == 4:
            geometric_model = 'affine_simple_4'
        elif output_size == 3:
            geometric_model = 'affine_simple'
        else:
            raise NotImplementedError('Geometric model deducted from output layer is unsupported')

        model = CNNGeometric(use_cuda=use_cuda,
                             output_dim=output_size,
                             **arg_groups['model'])

        if use_me is False:
            for name, param in model.FeatureExtraction.state_dict().items():
                if not name.endswith('num_batches_tracked'):
                    model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    

        for name, param in model.FeatureRegression.state_dict().items():
            if not name.endswith('num_batches_tracked'):
                model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])

        return (model,geometric_model)

    # Load model for stage 1
    model_1, geometric_model_1 = create_model(args.model_1)

    if two_stage:
        # Load model for stage 2
        model_2, geometric_model_2 = create_model(args.model_2)
    else:
        model_2,geometric_model_2 = None, None

    #import pdb; pdb.set_trace()

    print('Creating dataset and dataloader...')

    # Dataset and dataloader
    if args.eval_dataset == '3d' and use_me is False:
        cnn_image_size=(args.image_size,args.image_size)
        dataset = Dataset3D(csv_file = os.path.join(args.eval_dataset_path, 'all_pairs.csv'),
                      dataset_path = args.eval_dataset_path,
                      transform = NormalizeImageDict(['source_image','target_image']),
                      output_size = cnn_image_size)
        collate_fn = default_collate
    elif args.eval_dataset == '3d' and use_me is True:
        cnn_image_size=(args.input_height, args.input_width)
        dataset = Dataset3DME(csv_file = os.path.join(args.eval_dataset_path, 'all_pairs_3d.csv'),
                      dataset_path = args.eval_dataset_path,
                      input_size = cnn_image_size,
                      crop=args.crop_factor,
                      use_conf=args.use_conf)
        collate_fn = default_collate
    else:
        raise NotImplementedError('Dataset is unsupported')

    if use_cuda:
        batch_size = args.batch_size
    else:
        batch_size = 1

    dataloader = DataLoader(dataset, 
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers=0,
                            collate_fn = collate_fn)

    batch_tnf = BatchTensorToVars(use_cuda = use_cuda)

    if args.eval_dataset == '3d':
        metric = 'absdiff'
    else:
        raise NotImplementedError('Dataset is unsupported')
        
    model_1.eval()

    if two_stage:
        model_2.eval()

    print('Starting evaluation...')
        
    stats=compute_metric(metric,
                         model_1,
                         geometric_model_1,
                         model_2,
                         geometric_model_2,
                         dataset,
                         dataloader,
                         batch_tnf,
                         batch_size,
                         args)
    return stats 

if __name__ == '__main__':
    main()
