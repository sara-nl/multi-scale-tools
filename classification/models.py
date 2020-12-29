import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import time
import torch.nn.functional as F
import torch.utils.data
import os
import copy

class Single_Scale_Model(torch.nn.Module):
    def __init__(self,CNN_TO_USE, EMBEDDING_bool, DROPOUT, N_CLASSES):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Single_Scale_Model, self).__init__()

        self.CNN_TO_USE = CNN_TO_USE
        self.EMBEDDING_bool = EMBEDDING_bool
        self.DROPOUT = DROPOUT
        self.N_CLASSES = N_CLASSES

        pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', self.CNN_TO_USE, pretrained=True)

        if (('resnet' in self.CNN_TO_USE) or ('resnext' in self.CNN_TO_USE)):
            fc_input_features = pre_trained_network.fc.in_features
        elif (('densenet' in self.CNN_TO_USE)):
            fc_input_features = pre_trained_network.classifier.in_features
        elif ('mobilenet' in self.CNN_TO_USE):
            fc_input_features = pre_trained_network.classifier[1].in_features

        self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
        """
        if (torch.cuda.device_count()>1):
            self.conv_layers = torch.nn.DataParallel(self.conv_layers)
        """
        self.fc_feat_in = fc_input_features
        
        if (EMBEDDING_bool==True):

            if ('resnet34' in self.CNN_TO_USE):
                self.E = 128

            elif ('resnet50' in self.CNN_TO_USE):
                self.E = 256
            
            self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
            self.embedding_fc = torch.nn.Linear(in_features=self.E, out_features=self.N_CLASSES)

        else:
            self.fc = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.N_CLASSES)

    def forward(self, x):

        m_multiclass = torch.nn.Softmax()
        dropout = torch.nn.Dropout(p=self.DROPOUT)

        conv_layers_out=self.conv_layers(x)

        if ('densenet' in self.CNN_TO_USE):
            n = torch.nn.AdaptiveAvgPool2d((1,1))
            conv_layers_out = n(conv_layers_out)
        
        conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)
        
        if ('mobilenet' in self.CNN_TO_USE):
            dropout = torch.nn.Dropout(p=0.2)
            conv_layers_out = dropout(conv_layers_out)

        if (self.EMBEDDING_bool==True):
            embedding_layer = self.embedding(conv_layers_out)
            embedding_layer = dropout(embedding_layer)
            output_fcn = self.embedding_fc(embedding_layer)

            features_to_return = embedding_layer
        else:
            output_fcn = self.fc(conv_layers_out)
            features_to_return = conv_layers_out

        output_fcn = m_multiclass(output_fcn)
        output_fcn = torch.clamp(output_fcn, 1e-7, 1 - 1e-7)

        return output_fcn

class Multi_Scale_combine_probs(torch.nn.Module):
    def __init__(self, CNN_TO_USE, EMBEDDING_bool, DROPOUT, N_CLASSES, MAGNIFICATION, device):

        super(Multi_Scale_combine_probs, self).__init__()

        self.CNN_TO_USE = CNN_TO_USE
        self.EMBEDDING_bool = EMBEDDING_bool
        self.DROPOUT = DROPOUT
        self.N_CLASSES = N_CLASSES
        self.N_MAGNIFICATIONS = len(MAGNIFICATION)
        self.N_CLASSES = N_CLASSES
        self.device = device

        pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', self.CNN_TO_USE, pretrained=True)

        if (('resnet' in self.CNN_TO_USE) or ('resnext' in self.CNN_TO_USE)):
            fc_input_features = pre_trained_network.fc.in_features
        elif (('densenet' in self.CNN_TO_USE)):
            fc_input_features = pre_trained_network.classifier.in_features
        elif ('mobilenet' in self.CNN_TO_USE):
            fc_input_features = pre_trained_network.classifier[1].in_features

        self.fc_feat_in = fc_input_features
        feature_block = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
        
        if (EMBEDDING_bool==True):

            if ('resnet34' in self.CNN_TO_USE):
                self.E = 128
                self.L = self.E
                self.D = 64
                self.K = self.N_CLASSES

            elif ('resnet50' in self.CNN_TO_USE):
                self.E = 256
                self.L = self.E
                self.D = 128
                self.K = self.N_CLASSES

        else:

            if ('resnet34' in self.CNN_TO_USE):
                self.L = fc_input_features
                self.D = 128
                self.K = self.N_CLASSES

            elif ('resnet50' in self.CNN_TO_USE):
                self.L = self.E
                self.D = 256
                self.K = self.N_CLASSES

        features_extractors = [copy.deepcopy(feature_block) for i in range(self.N_MAGNIFICATIONS)]
        self.features_extractors = torch.nn.ModuleList(features_extractors)

        embeddings = [torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E) for i in range(self.N_MAGNIFICATIONS)]
        self.embeddings = torch.nn.ModuleList(embeddings)

        embeddings_fc = [torch.nn.Linear(in_features=self.E, out_features=self.N_CLASSES) for i in range(self.N_MAGNIFICATIONS)]
        self.embeddings_fc = torch.nn.ModuleList(embeddings_fc)

        self.classifier = torch.nn.Linear(in_features=self.N_CLASSES*self.N_MAGNIFICATIONS, out_features=self.N_CLASSES)

    def forward(self, x, idx_scale, mode_eval):

        #if used attention pooling
        #m = torch.nn.Softmax(dim=1)
        m_multiclass = torch.nn.Softmax()
        dropout = torch.nn.Dropout(p=self.DROPOUT)
        #print(conv_layers_out.shape)
        probs = []
        features = []

        if (mode_eval == 'multi_scale'):

            for i in range(len(self.features_extractors)):

                features_extractor = self.features_extractors[i]
                embedding = self.embeddings[i]
                embedding_fc = self.embeddings_fc[i]

                conv_layers_out=features_extractor(x[i].to(self.device))
                #print(x.shape)

                if ('densenet' in self.CNN_TO_USE):
                    n = torch.nn.AdaptiveAvgPool2d((1,1))
                    conv_layers_out = n(conv_layers_out)
                
                features_layer = conv_layers_out.view(-1, self.fc_feat_in)

                if ('mobilenet' in self.CNN_TO_USE):
                    dropout = torch.nn.Dropout(p=0.2)
                    features_layer = dropout(features_layer)
                #print(conv_layers_out.shape)

                features_layer = embedding(features_layer)
                features_layer = dropout(features_layer)
                out_embedding = embedding_fc(features_layer)
                features.append(out_embedding)
                prob_loc = m_multiclass(out_embedding)
                prob_loc = torch.clamp(prob_loc, 1e-7, 1 - 1e-7)

                probs.append(prob_loc)

            all_instances = torch.cat(features, dim=1)

            all_instances = all_instances.to(self.device)
            output_fcn = self.classifier(all_instances)
            output_fcn = m_multiclass(output_fcn)
            output_fcn = torch.clamp(output_fcn, 1e-7, 1 - 1e-7)

        elif (mode_eval == 'single_scale'):

            features_extractor = self.features_extractors[idx_scale]
            embedding = self.embeddings[idx_scale]
            embedding_fc = self.embeddings_fc[idx_scale]

            conv_layers_out=features_extractor(x.to(self.device))
            #print(x.shape)

            if ('densenet' in self.CNN_TO_USE):
                n = torch.nn.AdaptiveAvgPool2d((1,1))
                conv_layers_out = n(conv_layers_out)
            
            features_layer = conv_layers_out.view(-1, self.fc_feat_in)

            if ('mobilenet' in self.CNN_TO_USE):
                dropout = torch.nn.Dropout(p=0.2)
                features_layer = dropout(features_layer)
            #print(conv_layers_out.shape)

            features_layer = embedding(features_layer)
            features_layer = dropout(features_layer)
            out_embedding = embedding_fc(features_layer)
            features.append(out_embedding)
            output_fcn = m_multiclass(out_embedding)
            output_fcn = torch.clamp(output_fcn, 1e-7, 1 - 1e-7)

        return output_fcn, probs

class Multi_Scale_combine_features(torch.nn.Module):
    def __init__(self, CNN_TO_USE, EMBEDDING_bool, DROPOUT, N_CLASSES, MAGNIFICATION, device):

        super(Multi_Scale_combine_features, self).__init__()

        self.CNN_TO_USE = CNN_TO_USE
        self.EMBEDDING_bool = EMBEDDING_bool
        self.DROPOUT = DROPOUT
        self.N_CLASSES = N_CLASSES
        self.N_MAGNIFICATIONS = len(MAGNIFICATION)
        self.N_CLASSES = N_CLASSES
        self.device = device

        pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', self.CNN_TO_USE, pretrained=True)

        if (('resnet' in self.CNN_TO_USE) or ('resnext' in self.CNN_TO_USE)):
            fc_input_features = pre_trained_network.fc.in_features
        elif (('densenet' in self.CNN_TO_USE)):
            fc_input_features = pre_trained_network.classifier.in_features
        elif ('mobilenet' in self.CNN_TO_USE):
            fc_input_features = pre_trained_network.classifier[1].in_features

        self.fc_feat_in = fc_input_features
        feature_block = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
        
        if (EMBEDDING_bool==True):

            if ('resnet34' in self.CNN_TO_USE):
                self.E = 128
                self.L = self.E
                self.D = 64
                self.K = self.N_CLASSES

            elif ('resnet50' in self.CNN_TO_USE):
                self.E = 256
                self.L = self.E
                self.D = 128
                self.K = self.N_CLASSES

        else:

            if ('resnet34' in self.CNN_TO_USE):
                self.L = fc_input_features
                self.D = 128
                self.K = self.N_CLASSES

            elif ('resnet50' in self.CNN_TO_USE):
                self.L = self.E
                self.D = 256
                self.K = self.N_CLASSES

        features_extractors = [copy.deepcopy(feature_block) for i in range(self.N_MAGNIFICATIONS)]
        self.features_extractors = torch.nn.ModuleList(features_extractors)

        embeddings = [torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E) for i in range(self.N_MAGNIFICATIONS)]
        self.embeddings = torch.nn.ModuleList(embeddings)

        embeddings_fc = [torch.nn.Linear(in_features=self.E, out_features=self.N_CLASSES) for i in range(self.N_MAGNIFICATIONS)]
        self.embeddings_fc = torch.nn.ModuleList(embeddings_fc)

        self.classifier = torch.nn.Linear(in_features=self.E*self.N_MAGNIFICATIONS, out_features=self.N_CLASSES)

    def forward(self, x, idx_scale, mode_eval):

        #if used attention pooling
        #m = torch.nn.Softmax(dim=1)
        m_multiclass = torch.nn.Softmax()
        dropout = torch.nn.Dropout(p=self.DROPOUT)
        #print(conv_layers_out.shape)
        probs = []
        features = []

        if (mode_eval == 'multi_scale'):

            for i in range(len(self.features_extractors)):

                features_extractor = self.features_extractors[i]
                embedding = self.embeddings[i]
                embedding_fc = self.embeddings_fc[i]

                conv_layers_out=features_extractor(x[i].to(self.device))
                #print(x.shape)

                if ('densenet' in self.CNN_TO_USE):
                    n = torch.nn.AdaptiveAvgPool2d((1,1))
                    conv_layers_out = n(conv_layers_out)
                
                features_layer = conv_layers_out.view(-1, self.fc_feat_in)

                if ('mobilenet' in self.CNN_TO_USE):
                    dropout = torch.nn.Dropout(p=0.2)
                    features_layer = dropout(features_layer)
                #print(conv_layers_out.shape)

                features_layer = embedding(features_layer)
                features.append(features_layer)

                features_layer = dropout(features_layer)
                out_embedding = embedding_fc(features_layer)
                prob_loc = m_multiclass(out_embedding)
                prob_loc = torch.clamp(prob_loc, 1e-7, 1 - 1e-7)

                probs.append(prob_loc)

            all_instances = torch.cat(features, dim=1)

            all_instances = all_instances.to(self.device)
            output_fcn = self.classifier(all_instances)
            output_fcn = m_multiclass(output_fcn)
            output_fcn = torch.clamp(output_fcn, 1e-7, 1 - 1e-7)

        elif (mode_eval == 'single_scale'):

            features_extractor = self.features_extractors[idx_scale]
            embedding = self.embeddings[idx_scale]
            embedding_fc = self.embeddings_fc[idx_scale]

            conv_layers_out=features_extractor(x.to(self.device))
            #print(x.shape)

            if ('densenet' in self.CNN_TO_USE):
                n = torch.nn.AdaptiveAvgPool2d((1,1))
                conv_layers_out = n(conv_layers_out)
            
            features_layer = conv_layers_out.view(-1, self.fc_feat_in)

            if ('mobilenet' in self.CNN_TO_USE):
                dropout = torch.nn.Dropout(p=0.2)
                features_layer = dropout(features_layer)
            #print(conv_layers_out.shape)

            features_layer = embedding(features_layer)
            features.append(features_layer)

            features_layer = dropout(features_layer)
            out_embedding = embedding_fc(features_layer)
            output_fcn = m_multiclass(out_embedding)
            output_fcn = torch.clamp(output_fcn, 1e-7, 1 - 1e-7)

        return output_fcn, probs

if __name__ == "__main__":
	pass