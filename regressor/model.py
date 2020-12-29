import torch
from torch.utils import data
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data
import os
import random

class Regressor_model(torch.nn.Module):
    def __init__(self, CNN_TO_USE, EMBEDDING_bool, DROPOUT):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Regressor_model, self).__init__()

        pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', CNN_TO_USE, pretrained=True)

        if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
            fc_input_features = pre_trained_network.fc.in_features
        elif (('densenet' in CNN_TO_USE)):
            fc_input_features = pre_trained_network.classifier.in_features
        elif ('mobilenet' in CNN_TO_USE):
            fc_input_features = pre_trained_network.classifier[1].in_features

        self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
        """
        if (torch.cuda.device_count()>1):
            self.conv_layers = torch.nn.DataParallel(self.conv_layers)
        """
        self.fc_feat_in = fc_input_features
        self.EMBEDDING_bool = EMBEDDING_bool
        self.CNN_TO_USE = CNN_TO_USE
        self.OUTPUT_NETWORK = 1
        self.DROPOUT = DROPOUT
        if (EMBEDDING_bool==True):

            if ('resnet34' in CNN_TO_USE):
                self.E = 128

            elif ('resnet50' in CNN_TO_USE):
                self.E = 256
            
            self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
            self.embedding_fc = torch.nn.Linear(in_features=self.E, out_features=self.OUTPUT_NETWORK)

        else:
            self.fc = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.OUTPUT_NETWORK)


    def forward(self, x):

        #if used attention pooling
        output_att = None
        #m = torch.nn.Softmax(dim=1)
        dropout = torch.nn.Dropout(p=self.DROPOUT)
        #print(x.shape)
        conv_layers_out=self.conv_layers(x)
        #print(x.shape)

        if ('densenet' in self.CNN_TO_USE):
            n = torch.nn.AdaptiveAvgPool2d((1,1))
            conv_layers_out = n(conv_layers_out)
        
        conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)
        
        #print(conv_layers_out.shape)

        if ('mobilenet' in self.CNN_TO_USE):
            dropout = torch.nn.Dropout(p=0.2)
            conv_layers_out = dropout(conv_layers_out)
        #print(conv_layers_out.shape)

        if (self.EMBEDDING_bool==True):
            embedding_layer = self.embedding(conv_layers_out)
            embedding_layer = dropout(embedding_layer)
            output_fcn = self.embedding_fc(embedding_layer)

            features_to_return = embedding_layer
        else:
            output_fcn = self.fc(conv_layers_out)

        return output_fcn

if __name__ == "__main__":
	pass