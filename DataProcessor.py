import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import logging

from boxes_utils import *
from transformers import RobertaTokenizer, RobertaModel


class MyDataset(Data.Dataset):
    def __init__(self,data_dir,imagefeat_dir,tokenizer,max_seq_len=64,max_GT_boxes=1,num_roi_boxes=100,img_feat_dim=2048):
       
        
        self.imagefeat_dir=imagefeat_dir
        self.tokenizer=tokenizer
        self.relation_label_list=self.get_relation_labels()
        self.sentiment_label_list=self.get_sentiment_labels()
        self.max_seq_len=max_seq_len
        self.max_GT_boxes=max_GT_boxes
        self.examples=self.creat_examples(data_dir)
        self.number = len(self.examples)
        self.num_roi_boxes = num_roi_boxes
        self.img_feat_dim = img_feat_dim

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        return self.transform(line,index)   

    def creat_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for key,value in tqdm(dict.items(),desc="CreatExample"):
            examples.append(value)
        return examples

    def get_sentiment_labels(self):
        return ["0","1","2"]

    def get_relation_labels(self):
        return ["0","1"]

    def transform(self,line,index ):
        max_seq_len =self.max_seq_len
        max_GT_boxes =self.max_GT_boxes
        num_roi_boxes = self.num_roi_boxes

        value=line
        text_a = value['sentence'] 
        text_b = value['aspect']

        input_ids=self.tokenizer(text_a.lower(),text_b.lower())['input_ids']   #  <s>text_a</s></s>text_b</s>
        input_mask=[1]*len(input_ids)
        padding_id = [1]*(max_seq_len-len(input_ids)) #<pad> :1
        padding_mask=[0]*(max_seq_len-len(input_ids)) 

        input_ids += padding_id
        input_mask += padding_mask

       
        tokens=self.tokenizer.decode(input_ids)
        
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        
        
        img_id=0
        a_img_id = value['iid']
        a_GT_boxes=value['boxes']
        img_id=(a_img_id+str(' ')*(16-len(a_img_id))).encode('utf-8')
        GT_boxes=np.zeros((max_GT_boxes,4))
        if a_GT_boxes: 
            GT_boxes[:len(a_GT_boxes),:]=a_GT_boxes[:max_GT_boxes]
        
        roi_boxes=np.zeros((num_roi_boxes,4))  
        img_feat=np.zeros((num_roi_boxes,self.img_feat_dim))
        spatial_feat=np.zeros((num_roi_boxes,5))

        a_img_feat,a_spatial_feat,a_roi_boxes,img_shape=read_npz(self.imagefeat_dir,a_img_id) 
        assert len(a_roi_boxes)==len(a_img_feat)
        assert len(a_img_feat)==len(a_spatial_feat)
        current_num=len(a_img_feat)

        roi_boxes[:current_num,:]=a_roi_boxes
        img_feat[:current_num,:]=a_img_feat
        spatial_feat[:current_num,:]=a_spatial_feat


        box_label=get_attention_label(roi_boxes,GT_boxes) #box_label:[max_GT_boxes,NUM_ROI_BOXES] target:[max_GT_boxes,NUM_ROI_BOXES,4]

        relation_label=-1
        sentiment_label=-1
        sentiment_label_map = {label: i for i, label in enumerate(self.sentiment_label_list)}
        relation_label_map = {label : i for i, label in enumerate(self.relation_label_list)}
       
        rel=value['relation'] 
        if rel:
            if rel=='2':
                rel='1'
            relation_label=relation_label_map[rel]
        senti=value['sentiment']
        if senti:
            sentiment_label=sentiment_label_map[senti]
        
        
        return tokens,input_ids,input_mask, sentiment_label,img_id,img_shape,relation_label,GT_boxes,roi_boxes,img_feat,spatial_feat,box_label


    
def read_npz(imagefeat_dir,img_id):

    if 'twitter' in imagefeat_dir.lower():
        feat_dict=np.load(os.path.join(imagefeat_dir,img_id+'.jpg.npz'))
         
    img_feat=feat_dict['x']   #[2048,100]
    img_feat=img_feat.transpose((1,0))
    img_feat=(img_feat/np.sqrt((img_feat**2).sum()))

    # num_bbox=feat_dict['num_bbox']
            
    #boxes spatial
    bbox=feat_dict['bbox']
    img_h=feat_dict['image_h']
    img_w=feat_dict['image_w']
    spatial_feat=get_spatial_feat(bbox,img_h,img_w)  
        
    return img_feat,spatial_feat,bbox,[float(img_h),float(img_w)]

def get_spatial_feat(bbox,img_h,img_w):
     
    spatial_feat=np.zeros((bbox.shape[0],5),dtype=np.float)
    spatial_feat[:,0]=bbox[:,0]/float(img_w)
    spatial_feat[:,1]=bbox[:,1]/float(img_h)
    spatial_feat[:,2]=bbox[:,2]/float(img_w)
    spatial_feat[:,3]=bbox[:,3]/float(img_h)
    spatial_feat[:,4]=(bbox[:, 2] - bbox[:, 0])*(bbox[:, 3] - bbox[:, 1]) / float(img_h*img_w)
    return spatial_feat

