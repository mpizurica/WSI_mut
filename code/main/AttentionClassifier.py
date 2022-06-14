# from https://github.com/huiqu18/GeneMutationFromHE/blob/main/code_prediction_on_breast_cancer/models.py

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

# from https://github.com/huiqu18/GeneMutationFromHE/blob/main/code_data_processing/3-2.feature_extraction.py
class ResNet_extractor(nn.Module):
    def __init__(self, layers=101):
        super().__init__()
        self.layers = layers
        if layers == 18:
            self.resnet = models.resnet18(pretrained=True)
        elif layers == 34:
            self.resnet = models.resnet34(pretrained=True)
        elif layers == 50:
            self.resnet = models.resnet50(pretrained=True)
        elif layers == 101:
            self.resnet = models.resnet101(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        if self.layers == 18:
            x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        

# adapted from https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, D2 = 128, dropout = None, use_b = True, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()

        self.fc = [nn.Linear(L, D), nn.ReLU()]
        self.use_b = use_b

        if dropout != None:
            self.fc.append(nn.Dropout(dropout))

        self.attention_a = [
            nn.Linear(D, D2),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(D, D2),
                            nn.Sigmoid()]
        if dropout != None:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.fc = nn.Sequential(*self.fc)
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D2, n_classes)

        self.classifiers = nn.Linear(D, n_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

    def forward(self, x, attn_mask):
        # input feature size of x = [batch_size, n_patch, feature_dim]

        # transform features
        x = self.fc(x) #[batch_size, n_patch, hidden_dim]

        # calculate attention weights
        a = self.attention_a(x) #[batch_size, n_patch, hidden_dim2]

        if self.use_b:
            b = self.attention_b(x) #[batch_size, n_patch, hidden_dim2]
            A = torch.mul(a, b) #[batch_size, n_patch, hidden_dim2]

        else:
            A = a

        A = self.attention_c(A)  # batch_size x n_patch x n_classes 

        A = torch.transpose(A, 1, 2) # batch_size x n_classes x n_patch 
        A_masked = A+attn_mask.reshape(A.shape)
        A = F.softmax(A_masked, dim=2) # batch_size x n_classes x n_patch 

        # weighted sum of patches according to attention weights
        M = torch.bmm(A, x) # batch_size x n_classes x hidden_dim
        logits = self.classifiers(M) # batch_size x n_classes x n_classes

        return logits, A