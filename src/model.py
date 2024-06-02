import torch 
from torch import nn
import copy
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, EfficientNet_B0_Weights, efficientnet_b0
 
class squarePartition(nn.Module):
    def __init__(self, HW, shift=0, device='cpu') -> None:
        super(squarePartition, self).__init__()

        assert (HW > 0 and HW % 8 == 0)

        self.shift = shift
        self.HW = HW
        self.part = HW//8
        self.center = HW//2


        self._bp_tensor = self._create_base_partitions()
        self._bp_tensor.requires_grad = False
        self._bp_tensor = self._bp_tensor.to(device)


    def _create_base_partitions(self):
        result = []

        zeros = torch.zeros(size=(1, 1, self.HW, self.HW))
        ones = torch.ones(size=(1, 1, self.HW, self.HW))

        for i in range(1, 5):
            

            d1 = min(max(self.center - i*self.part + self.shift*self.part, 0), self.HW)
            d2 = max(min(self.center + i*self.part + self.shift*self.part, self.HW), 0)

            
            if i != 4:
                partition = copy.deepcopy(zeros)
                partition[:,:,d1:min(d1+self.part, self.HW), d1:d2] = 1
                partition[:,:,max(d2-self.part,0):d2, d1:d2] = 1
                partition[:,:,d1:d2, d1:min(d1+self.part, self.HW)] = 1
                partition[:,:,d1:d2, max(d2-self.part,0):d2] = 1
            else:
                partition = copy.deepcopy(ones)
                partition -= result[-1]
                partition -= result[-2]
                partition -= result[-3]

            
            result.append(partition)

        result.append(result[0]+result[1])
        result.append(result[1]+result[2])
        result.append(result[2]+result[3])
        result.append(result[4]+result[2])
        result.append(result[5]+result[3])
        result.append(result[7]+result[3])

        bp_tensor = torch.Tensor(size=(len(result), 1, self.HW, self.HW))
        torch.cat(result, out=bp_tensor, dim=0)
        return bp_tensor


    def forward(self, x):
        result = x.unsqueeze(1)*self._bp_tensor.unsqueeze(0)
        return result
    

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
class weightEstimator(nn.Module):
    def __init__(self, in_channels, activation=nn.ReLU()):
        super(weightEstimator, self).__init__()
        self.activation = activation
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            self.activation,
        )
        self.gem_pooling = GeM()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.Linear(512, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.gem_pooling(x)
        B, C, _, _ = x.shape
        x = x.reshape(B, C)
        x = self.fc(x)
        return x

class SDP_Classifier(nn.Module):
    def __init__(self, in_channels=2048, num_classes=200, prob=0.2):
        super(SDP_Classifier, self).__init__()
        
        self.feature_compression = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=prob)
        )
        self.cls = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.feature_compression(x)
        x = self.cls(x)
        return x


class ShiftingDensePartition(nn.Module):
    def __init__(self, in_channels=1024, HW=32, p=3, eps=1e-6, w_act = nn.ReLU(), device='cpu'):
        super(ShiftingDensePartition, self).__init__()

        self.square_partition1 = squarePartition(HW=HW, shift=0, device=device)
        self.square_partition2 = squarePartition(HW=HW, shift=1, device=device)
        self.square_partition3 = squarePartition(HW=HW, shift=-1, device=device)

        self.gem_pooling = GeM(p=p, eps=eps)

        self.weight_estimator = weightEstimator(in_channels=in_channels, activation=w_act)


    def forward(self, x):
        partitions1 = self.square_partition1(x)
        partitions2 = self.square_partition2(x)
        partitions3 = self.square_partition3(x)

        weights = self.weight_estimator(x)

        B, P, C, H, W = partitions1.shape
        _, D = weights.shape


        partitions1 = self.gem_pooling(partitions1.reshape(B, -1, H, W))
        partitions2 = self.gem_pooling(partitions2.reshape(B, -1, H, W))
        partitions3 = self.gem_pooling(partitions3.reshape(B, -1, H, W))

        partitions1 = partitions1.reshape(B, P, C).unsqueeze(1)
        partitions2 = partitions2.reshape(B, P, C).unsqueeze(1)
        partitions3 = partitions3.reshape(B, P, C).unsqueeze(1)

        part_t = torch.cat((partitions1, partitions2, partitions3), dim=1)
        
        part_t = part_t.permute(0, 3, 2, 1).reshape(B, -1, D)
        weights = weights.unsqueeze(-1)

        weighted_output = torch.matmul(part_t, weights)
        weighted_output = weighted_output.reshape(B, C, P).permute(0, 2, 1)

        return weighted_output

class ResNet50_backbone(nn.Module):
    def __init__(self, in_hw = 512):
        super(ResNet50_backbone, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.hw = in_hw//16
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        # Modify conv2 and downsample stride at conv5_1 from 2 to 1
        self.resnet.layer4[0].conv2.stride = (1,1)
        self.resnet.layer4[0].downsample[0].stride = (1,1)

    def forward(self, x):
        return self.resnet(x).reshape(-1, 2048, self.hw, self.hw)
    
class EfficientNet_backbone(nn.Module):
    def __init__(self, in_hw = 512):
        super(EfficientNet_backbone, self).__init__()
        self.effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.hw = in_hw//32
        self.effnet.avgpool = nn.Identity()
        self.effnet.classifier = nn.Identity()

    def forward(self, x):
        return self.effnet(x).reshape(-1, 1280, self.hw, self.hw)


class SDP_CrossView_model(nn.Module):
    def __init__(self, 
                 backbone = 'resnet', 
                 in_hw = 512,
                 blocks = 10,
                 num_classes = 200, 
                 weight_estimator_activation=nn.ReLU(),
                 device='cpu'):
        super(SDP_CrossView_model, self).__init__()
        self.num_classes = num_classes
        self.blocks = blocks
        if backbone == 'resnet':
            self.backbone = ResNet50_backbone(in_hw=in_hw)
            self.channels = 2048
            self.d = 16
        else:
            self.backbone = EfficientNet_backbone(in_hw=in_hw)
            self.channels = 1280
            self.d = 32

        self.do_partitions = ShiftingDensePartition(in_channels=self.channels, HW=in_hw//self.d, w_act=weight_estimator_activation, device=device)
        
        for i in range(1, self.blocks+1):
            setattr(self, f'classifier{i}', SDP_Classifier(in_channels=self.channels, num_classes=self.num_classes))
        #self.classifier = SDP_Classifier(in_channels=self.channels, num_classes=self.num_classes)
        
    def part_classifier(self, x, mode):
        part = {}
        predict = []
        for i in range(self.blocks):
            part[i] = x[:, i, :].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            cls = getattr(self, f'classifier{i+1}')
            if (mode == 'retrieve'):
                predict.append(cls.feature_compression(part[i]))
            else:
                predict.append(cls(part[i]))
         
        return torch.stack(predict, dim=1)

    def forward(self, x, mode = 'cls'):
        x = self.backbone(x)
        x = self.do_partitions(x)
        x = self.part_classifier(x, mode)
        return x


class SDPL_ClsLoss(nn.Module):
    def __init__(self, blocks=10):
        super(SDPL_ClsLoss, self).__init__()
        self.blocks=10
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, 
                drone_out, 
                drone_gt, 
                satellite_out, 
                satellite_gt):
        
        #drone_out = self.softmax(drone_out)
        #sattelite_out = self.softmax(sattelite_out)
        result = 0
        for i in range(self.blocks):
            d_part = drone_out[:,i,:].view(drone_out.size(0), -1)
            s_part = satellite_out[:,i,:].view(satellite_out.size(0), -1)
            result += (
                self.loss(d_part,drone_gt)
                + self.loss(s_part,satellite_gt)
            )
            
        return result





        