from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.matlib
from geotnf.transformation import GeometricTnf

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

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out

class MERegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True, batch_normalization=True, channels=[4,32,32,64,64,128]):
        super(MERegression, self).__init__()
        nn_modules = []

        num_blocks = len(channels) - 1
        input_size = np.array([216,384], dtype=int)

        for i in range(num_blocks):
            nn_modules.append(BasicBlock(channels[i], channels[i+1], batch_normalization=batch_normalization))
            nn_modules.append(nn.MaxPool2d(kernel_size=2))
            input_size //= 2
        self.conv = nn.Sequential(*nn_modules)    

        self.linear = nn.Linear(channels[-1] * input_size[0] * input_size[1], output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x
    
    
class CNNGeometric(nn.Module):
    def __init__(self, output_dim=6, 
                 feature_extraction_cnn='vgg', 
                 feature_extraction_last_layer='',
                 fr_kernel_sizes=[7,5,5],
                 fr_channels=[225,128,64],
                 normalize_matches=True, 
                 use_me=False,
                 batch_normalization=True, 
                 use_cuda=True,
                 matching_type='correlation'):
        
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_matches = normalize_matches
        self.use_me = use_me
        if self.use_me:
            self.FeatureRegression = MERegression(output_dim,
                                             use_cuda=self.use_cuda,
                                             channels=fr_channels,
                                             batch_normalization=batch_normalization)
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
            mv_L2R = tnf_batch['mv_L2R']
            mv_R2L = tnf_batch['mv_R2L']
            mv_concat = torch.cat((mv_L2R, -mv_R2L), dim=1)
            theta = self.FeatureRegression(mv_concat)
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