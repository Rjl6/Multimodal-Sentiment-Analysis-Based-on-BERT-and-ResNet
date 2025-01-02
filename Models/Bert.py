import torch
import torch.nn as nn
from transformers import AutoModel


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.text_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        
        # 是否进行fine-tune
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor(config.loss_weight))
        
        # self.bert.init_weights()

    def forward(self, texts, texts_mask,imgs, labels = None):
        bert_out = self.bert(input_ids=texts, token_type_ids=None, attention_mask=texts_mask)
        pooler_out = bert_out['pooler_output']
        prob_vec = self.text_classifier(pooler_out)
        pred_labels = torch.argmax(prob_vec,dim = 1)
        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels
