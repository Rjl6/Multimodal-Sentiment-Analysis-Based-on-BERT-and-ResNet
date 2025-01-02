import torch
import torch.nn as nn
from torchvision.models import resnet50

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(
            *(list(self.full_resnet.children())[:-1]),
            nn.Flatten()
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # 是否进行fine-tune
        for param in self.full_resnet.parameters():
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.img_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor(config.loss_weight))

    def forward(self,texts, texts_mask,imgs, labels = None):
        feature = self.resnet(imgs)
        
        img_feature = self.trans(feature)
        prob_vec = self.img_classifier(img_feature)
        
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels