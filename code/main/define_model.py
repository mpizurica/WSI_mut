import torch.nn as nn
import torchvision.models as models

from AttentionClassifier import Attn_Net_Gated


def get_final_model(model_name, attention_module, hidden_dim, hidden_dim2, use_b, dropout):
    
    if attention_module != None:
        if attention_module == 'faisal':

            if model_name == 'RESNET18':
                dim = 512
            elif model_name == 'RESNET50':
                dim = 1024
            else:
                print('wrong model name')
                exit()
            
            model = Attn_Net_Gated(dim,hidden_dim,hidden_dim2,dropout,use_b,1)
        else:
            exit() # not running the other anymore

    else:
        if model_name == 'RESNET18':
            model = get_model(model_name)
            
        else:
            print('Wrong model name')
            exit()
            
    model = model.cuda()       
    return model

def get_model(model_name):

    if model_name == 'RESNET18':
        
        model = models.resnet18(pretrained=True) 
        for param in model.parameters():
            param.requires_grad = False

        fc = nn.Linear(in_features=512, out_features=1, bias=True)
        model.fc = fc

        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == 'RESNET50':
        model = models.resnet50(pretrained=True) 
        for param in model.parameters():
            param.requires_grad = False

        fc = nn.Linear(in_features=2048, out_features=1, bias=True)
        model.fc = fc

        for param in model.fc.parameters():
            param.requires_grad = True

    return model