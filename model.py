import torch
import torch.nn as nn
import torchvision

class ConvNet(nn.Module):
    def __init__(self,model,num_classes):
        super(ConvNet,self).__init__()
        self.base_model = nn.Sequential(*list(model.children())[:-1]) # model excluding last FC layer
        self.linear1 = nn.Linear(in_features=62720,out_features=4096) # flattened dimension of mobilenet_v2 
        self.linear2 = nn.Linear(in_features=4096,out_features=256)
        self.linear3 = nn.Linear(in_features=256,out_features=num_classes)
        self.relu = nn.LeakyReLU()
    
    def forward(self,x):
        x = self.base_model(x)
        x = torch.flatten(x,1)
        # print(x.shape)
        x = self.linear1(x)
        x = self.relu(x)
        lin = self.linear2(x)
        x = self.relu(lin)
        x = self.linear3(x)
        return lin, x


def get_model(device, num_classes):
	model = torchvision.models.mobilenet_v2(pretrained=True)
	model = model.to(device)
	model = ConvNet(model, num_classes)
	model = model.to(device)
	return model


def test_model(model, device):
    # testing if model has any ambiguity
    vec = torch.ones((4,3,224,224),dtype=torch.float32)
    vec = vec.to(device)
    feature, out = model(vec)
    print(feature.shape, out.shape)