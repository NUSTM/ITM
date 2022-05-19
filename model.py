import copy
import math
import sys

import torch
from torch import nn, reshape
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from modeling_utils import BertSelfEncoder, BertCrossEncoder_AttnMap, BertPooler, BertLayerNorm
import torch.nn.functional as F

from transformers import  RobertaModel, AutoConfig

import logging
logger = logging.getLogger(__name__)



class Coarse2Fine(nn.Module):
    def __init__(self,roberta_name='roberta-base',img_feat_dim=2048,roi_num=100):
        super().__init__()
        self.img_feat_dim = img_feat_dim
        config = AutoConfig.from_pretrained(roberta_name)
        self.hidden_dim = config.hidden_size

        self.roberta = RobertaModel.from_pretrained(roberta_name)
        self.sent_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.feat_linear = nn.Linear(self.img_feat_dim, self.hidden_dim)
        self.img_self_attn = BertSelfEncoder(config, layer_num=1)
        
        self.v2t=BertCrossEncoder_AttnMap(config, layer_num=1)
        
        self.dropout1=nn.Dropout(0.3)
        self.gather=nn.Linear(self.hidden_dim,1)
        self.dropout2=nn.Dropout(0.3)
        self.pred=nn.Linear(roi_num,2)
        self.ce_loss=nn.CrossEntropyLoss()

        self.t2v=BertCrossEncoder_AttnMap(config,layer_num=1)
        self.ranking_loss = nn.KLDivLoss(reduction='batchmean')
        
        self.senti_selfattn = BertSelfEncoder(config, layer_num=1)
        self.first_pooler = BertPooler(config)
        self.senti_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.senti_detc = nn.Linear(self.hidden_dim, 3)
        
        self.init_weight()
    


    def init_weight(self):
        ''' bert init
        '''
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name ): #linear/embedding
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name ):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name ):
                module.bias.data.zero_()


    def forward(self, img_id, 
            input_ids,  input_mask, img_feat, 
            relation_label,box_labels,
            ranking_loss_ratio=1.,pred_loss_ratio=1.):
        # input_ids,input_mask : [N, L]
        #             img_feat : [N, 100, 2048]
        #         spatial_feat : [N, 100, 5]
        #            box_label : [N, 1, 100 ]
        # box_labels: if IoU > 0.5, IoU_i/sum(); 0
        

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size, seq = input_ids.size()
        _, roi_num, feat_dim = img_feat.size()  # =100

        
      
        # text feature
        roberta_output=self.roberta(input_ids,input_mask)
        sentence_output=roberta_output.last_hidden_state
        text_pooled_output=roberta_output.pooler_output


        #sentence_output = self.sent_dropout(sentence_output)

        
        # visual self Attention
        img_feat_ = self.feat_linear(img_feat)  # [N*n, 100, 2048] ->[N*n, 100, 768] 
        image_mask = torch.ones((batch_size, roi_num)).to(device)
        extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)
        extended_image_mask = extended_image_mask.to(dtype=next(self.parameters()).dtype)
        extended_image_mask = (1.0 - extended_image_mask) * -10000.0  
        visual_output = self.img_self_attn(img_feat_, extended_image_mask)          #image self atttention
        visual_output = visual_output[-1]  # [N*n, 100, 768]
       
        
        #1. visual query sentence : 
        extended_sent_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_sent_mask = extended_sent_mask.to(dtype=next(self.parameters()).dtype)
        extended_sent_mask = (1.0 - extended_sent_mask) * -10000.0
        sentence_aware_image,_=self.v2t(visual_output,
                                        sentence_output,
                                        extended_sent_mask,
                                        output_all_encoded_layers=False)  # image query sentence
        sentence_aware_image=sentence_aware_image[-1]  #[N,100,768]

        gathered_sentence_aware_image=self.gather(self.dropout1( 
                                                        sentence_aware_image)).squeeze(2) #[N,100,768]->[N,100,1] ->[N,100]
        rel_pred=self.pred(self.dropout2(
                                gathered_sentence_aware_image)) #  [N,2]
        
        gate=torch.softmax(rel_pred,dim=-1)[:,1].unsqueeze(1).expand(batch_size,
                                                                    roi_num).unsqueeze(2) .expand(batch_size,
                                                                                                 roi_num,self.hidden_dim)
        
        gated_sentence_aware_image = gate * sentence_aware_image

        
        #2.cls query gated_visual: Ranking
        image_aware_sentence,Attn_map=self.t2v(text_pooled_output.unsqueeze(1),
                                               gated_sentence_aware_image,
                                               extended_image_mask)          #[N,1,768]   
        image_aware_sentence=image_aware_sentence[-1] #[N,1,768]
        Attn_map=Attn_map[-1] #[N,1,100]
        
        ## ranking loss: kl_div
        if relation_label!= None and box_labels !=None:
            box_labels = box_labels.reshape(-1, roi_num)  # [N*n,100]
            pred_loss=self.ce_loss(rel_pred,relation_label.long())
            ranking_loss = self.ranking_loss(F.softmax(Attn_map.squeeze(1), dim=1).log(), box_labels)   #box_label :soft label   #0.7927        
        else:
            pred_loss=torch.tensor(0.,requires_grad=True).to(device)
            ranking_loss=torch.tensor(0.,requires_grad=True).to(device)


        #sentiment classifier
        senti_mixed_feature=torch.cat((image_aware_sentence,sentence_output),dim=1) #[N,1+L,756]
        senti_mask = torch.ones((batch_size, 1)).to(device)
        senti_mask = torch.cat((senti_mask, input_mask), dim=-1).to(device)
        extended_senti_mask = senti_mask.unsqueeze(1).unsqueeze(2)
        extended_senti_mask = extended_senti_mask.to(dtype=next(self.parameters()).dtype)
        extended_senti_mask = (1.0 - extended_senti_mask) * -10000.0
        senti_mixed_output = self.senti_selfattn(senti_mixed_feature, extended_senti_mask)  # [N, L+1, 768]
        senti_mixed_output = senti_mixed_output[-1]

        senti_comb_img_output = self.first_pooler(senti_mixed_output)
        senti_pooled_output = self.senti_dropout(senti_comb_img_output)
        senti_pred = self.senti_detc(senti_pooled_output)
        
        
        return senti_pred,ranking_loss_ratio*ranking_loss,pred_loss_ratio*pred_loss,rel_pred,Attn_map

