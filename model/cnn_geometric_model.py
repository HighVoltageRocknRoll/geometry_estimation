from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
try:
    import torchvision.models as models
except:
    pass
import numpy as np
import numpy.matlib
from geotnf.transformation import GeometricTnf
from .resnet import myresnet, myresnet_big

def featureL2Norm(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)

class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='vgg', normalization=True, last_layer='', use_cuda=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                         'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                         'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                         'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                         'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']
            if last_layer=='':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx+1])
        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer=='':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]
            
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])
        if feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to denseblock3
            # self.model = nn.Sequential(*list(self.model.features.children())[:-3])
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(*list(self.model.features.children())[:-4])
        if not train_fe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()
        
    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = featureL2Norm(features)
        return features
    
class FeatureCorrelation(torch.nn.Module):
    def __init__(self,shape='3D',normalization=True,matching_type='correlation'):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.matching_type=matching_type
        self.shape=shape
        self.ReLU = nn.ReLU()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        if self.matching_type=='correlation':
            if self.shape=='3D':
                # reshape features for matrix multiplication
                feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
                feature_B = feature_B.view(b,c,h*w).transpose(1,2)
                # perform matrix mult.
                feature_mul = torch.bmm(feature_B,feature_A)
                # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
            elif self.shape=='4D':
                # reshape features for matrix multiplication
                feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
                feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
                # perform matrix mult.
                feature_mul = torch.bmm(feature_A,feature_B)
                # indexed [batch,row_A,col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b,h,w,h,w).unsqueeze(1)
            
            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
        
            return correlation_tensor

        if self.matching_type=='subtraction':
            return feature_A.sub(feature_B)
        
        if self.matching_type=='concatenation':
            return torch.cat((feature_A,feature_B),1)

class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True, batch_normalization=True, kernel_sizes=[7,5,5], channels=[225,128,64]):
        super(FeatureRegression, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers-1): # last layer is linear 
            k_size = kernel_sizes[i]
            ch_in = channels[i]
            ch_out = channels[i+1]            
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)        
        self.linear = nn.Linear(ch_out * kernel_sizes[-1] * kernel_sizes[-1], output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, batch_normalization=True):
        super(BasicBlock, self).__init__()
        
        layers = []

        layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, padding=1))
        if batch_normalization:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1))
        if batch_normalization:
            layers.append(nn.BatchNorm2d(planes))

        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

        if inplanes != planes:
            downsample_layers = []
            downsample_layers.append(nn.Conv2d(inplanes, planes, kernel_size=1))
            if batch_normalization:
                downsample_layers.append(nn.BatchNorm2d(planes))
            self.downsample = nn.Sequential(*downsample_layers)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class MERegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True, batch_normalization=True, channels=[4,32,32,64,64,128], extended_prep_layer=True):
        super(MERegression, self).__init__()

        layers = []
        layers.append(nn.Conv2d(channels[0], channels[1], kernel_size=7, stride=2, padding=3, bias=(not extended_prep_layer)))
        if batch_normalization:
            layers.append(nn.BatchNorm2d(channels[1]))
        if extended_prep_layer:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.prep_layer = nn.Sequential(*layers)

        channels = channels[1:]
        nn_modules = []
        num_blocks = len(channels) - 1

        for i in range(num_blocks):
            nn_modules.append(BasicBlock(channels[i], channels[i+1], batch_normalization=batch_normalization))
        self.residual_layers = nn.Sequential(*nn_modules)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(channels[-1], output_dim)

        if use_cuda:
            self.prep_layer.cuda()
            self.residual_layers.cuda()
            self.avgpool.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.residual_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

class MERegressionResnet(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True, batch_normalization=True, channels=[6,2,2,2,2]):
        super(MERegressionResnet, self).__init__()
        self.resnet = myresnet(output_dim=output_dim, batch_normalization=batch_normalization, channels=channels)
        if use_cuda:
            self.resnet = self.resnet.cuda()

    def forward(self, x):
        x = self.resnet(x)
        return x

class MERegressionResnetBig(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True, batch_normalization=True, channels=[6,2,2,2,2]):
        super(MERegressionResnetBig, self).__init__()
        self.resnet = myresnet_big(output_dim=output_dim, batch_normalization=batch_normalization, channels=channels)
        if use_cuda:
            self.resnet = self.resnet.cuda()

    def forward(self, x):
        x = self.resnet(x)
        return x

class CNNGeometric(nn.Module):
    def __init__(self, output_dim=6, 
                 feature_extraction_cnn='vgg', 
                 feature_extraction_last_layer='',
                 fr_kernel_sizes=[7,5,5],
                 fr_channels=[225,128,64],
                 normalize_matches=True, 
                 use_me=False,
                 use_siamese=False,
                 extended_prep_layer=True,
                 me_main_input='disparity',
                 use_backward_input=False,
                 use_conf=False,
                 grid_input=False,
                 batch_normalization=True, 
                 use_cuda=True,
                 matching_type='correlation'):
        
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_matches = normalize_matches
        self.use_me = use_me
        self.use_siamese = use_siamese
        self.theta_inverter = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32, requires_grad=False)
        if self.use_cuda:
            self.theta_inverter = self.theta_inverter.cuda()

        if self.use_me:
            self.model_input_keys = []
            self.siamese_input_keys = []

            if me_main_input == 'disparity' or me_main_input == 'both':
                self.model_input_keys.append('mv_L2R')
                self.siamese_input_keys.append('mv_R2L')
                if use_backward_input:
                    self.model_input_keys.append('mv_R2L')
                    self.siamese_input_keys.append('mv_L2R')

            if me_main_input == 'grid' or me_main_input == 'both':
                self.model_input_keys.append('grid_L2R')
                self.siamese_input_keys.append('grid_R2L')
                if use_backward_input:
                    self.model_input_keys.append('grid_R2L')
                    self.siamese_input_keys.append('grid_L2R')

            if use_conf:
                self.model_input_keys.append('conf_L')
                self.siamese_input_keys.append('conf_R')
                if use_backward_input:
                    self.model_input_keys.append('conf_R')
                    self.siamese_input_keys.append('conf_L')

            if grid_input:
                self.model_input_keys.append('grid')
                self.siamese_input_keys.append('grid')
            
            if feature_extraction_cnn == 'resnet_big':
                print('Using ResNet with bottleneck arch')
                self.FeatureRegression = MERegressionResnetBig(output_dim,
                                             use_cuda=self.use_cuda,
                                             channels=fr_channels,
                                             batch_normalization=batch_normalization)
                
            elif feature_extraction_cnn == 'resnet':
                print('Using ResNet arch')
                self.FeatureRegression = MERegressionResnet(output_dim,
                                             use_cuda=self.use_cuda,
                                             channels=fr_channels,
                                             batch_normalization=batch_normalization)
            else:
                print('Using my MERegression arch')
                self.FeatureRegression = MERegression(output_dim,
                                             use_cuda=self.use_cuda,
                                             channels=fr_channels,
                                             batch_normalization=batch_normalization,
                                             extended_prep_layer=extended_prep_layer)
            if self.use_cuda:
                self.FeatureRegression = self.FeatureRegression.cuda()
        else:    
            self.FeatureExtraction = FeatureExtraction(train_fe=False,
                                                    feature_extraction_cnn=feature_extraction_cnn,
                                                    last_layer=feature_extraction_last_layer,
                                                    normalization=True,
                                                    use_cuda=self.use_cuda)
            
            self.FeatureCorrelation = FeatureCorrelation(shape='3D',normalization=normalize_matches,matching_type=matching_type)        
            

            self.FeatureRegression = FeatureRegression(output_dim,
                                                    use_cuda=self.use_cuda,
                                                    kernel_sizes=fr_kernel_sizes,
                                                    channels=fr_channels,
                                                    batch_normalization=batch_normalization)


    # used only for foward pass at eval and for training with strong supervision
    def forward(self, tnf_batch):
        if self.use_me:
            model_input = []
            for key in self.model_input_keys:
                model_input.append(tnf_batch[key])

            model_input = torch.cat(model_input, dim=1)
            theta = self.FeatureRegression(model_input)

            if self.use_siamese:
                siamese_input = []
                for key in self.siamese_input_keys:
                    siamese_input.append(tnf_batch[key])
                siamese_input = torch.cat(siamese_input, dim=1)
                theta_siamese = self.FeatureRegression(siamese_input)
                theta = torch.cat([theta, self.theta_inverter - theta_siamese], dim=1)
            
            return theta
            
        else:
            # feature extraction
            feature_A = self.FeatureExtraction(tnf_batch['source_image'])
            feature_B = self.FeatureExtraction(tnf_batch['target_image'])
            # feature correlation
            correlation = self.FeatureCorrelation(feature_A,feature_B)
            # regression to tnf parameters theta
            theta = self.FeatureRegression(correlation)
            
            return theta