import os
import logging
import argparse
import random
import cv2
import datetime
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import precision_recall_fscore_support
from torch import optim
from torch.nn import CrossEntropyLoss

from DataProcessor import *
from model import Coarse2Fine
from boxes_utils import*
from optimization import BertAdam



def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def post_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokens,input_ids,input_mask, sentiment_label,\
                        img_id,img_shape,relation_label, GT_boxes,roi_boxes,img_feat,spatial_feat,box_labels=batch 
        
    input_ids=list(map(list, zip(*input_ids)))
    input_mask=list(map(list, zip(*input_mask)))
    img_shape=list(map(list, zip(*img_shape)))
    
        
    input_ids=torch.tensor(input_ids,dtype=torch.long).to(device)
    input_mask=torch.tensor(input_mask,dtype=torch.long).to(device)
    img_shape=torch.tensor(img_shape,dtype=torch.float).to(device)
    sentiment_label=sentiment_label.to(device).long()
    relation_label=relation_label.to(device).long()
    GT_boxes=GT_boxes.to(device).float()
    roi_boxes=roi_boxes.to(device).float()
    img_feat=img_feat.to(device).float()
    spatial_feat=spatial_feat.to(device).float()
    box_labels=box_labels.to(device).float()
   
            
    return tokens,input_ids,input_mask, sentiment_label,\
                        img_id,img_shape,relation_label, GT_boxes,roi_boxes,img_feat,spatial_feat,box_labels



def inference_sentiment(model, test_dataloader, output_dir, logger):
    model.eval()
    nb_eval_examples = 0
    test_senti_acc=0
    senti_true_label_list = []
    senti_pred_label_list = []
    img_id_list=[]
    
    for batch in tqdm(test_dataloader, desc="Testing_SA"):
        tokens,input_ids,input_mask, sentiment_label,\
                    img_id,img_shape,relation_label, GT_boxes,roi_boxes,img_feat,spatial_feat,box_labels=post_dataloader(batch)

        with torch.no_grad():
            #senti_pred = model(img_id=img_id,
            senti_pred,ranking_loss,pred_loss,pred_score,attn_map= model(img_id=img_id,
                                                input_ids     = input_ids, 
                                                input_mask   = input_mask, 
                                                img_feat     = img_feat, 
                                                relation_label    = None,
                                                box_labels = None
                                                )
            
        sentiment_label=sentiment_label.cpu().numpy()
        senti_pred=senti_pred.cpu().numpy()
        senti_true_label_list.append(sentiment_label)
        senti_pred_label_list.append(senti_pred)
        img_id_list.append(img_id)
        tmp_senti_accuracy = accuracy(senti_pred, sentiment_label)
        test_senti_acc += tmp_senti_accuracy
    
            
        current_batch_size=input_ids.size()[0]
        nb_eval_examples += current_batch_size


    test_senti_acc = test_senti_acc / nb_eval_examples
    
    senti_true_label = np.concatenate(senti_true_label_list)
    senti_pred_outputs = np.concatenate(senti_pred_label_list)
    img_ids = np.concatenate(img_id_list)
    test_senti_precision, test_senti_recall, test_senti_F_score = macro_f1(senti_true_label, senti_pred_outputs)

    result = {
            'Test_senti_acc':test_senti_acc,
            'Test_senti_precision':test_senti_precision,
            'Test_senti_recall':test_senti_recall,
            'Test_senti_F_score':test_senti_F_score}
    logger.info("***** Test Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    pred_label = np.argmax(senti_pred_outputs, axis=-1)
    assert len(pred_label)==len(img_ids)
    assert len(img_ids)==len(senti_true_label)
    assert len(senti_true_label)==nb_eval_examples

    fout_p = open(os.path.join(output_dir, "sentiment_pred.txt"), 'w')
    fout_p.write("ground truth ---- pred label ---- img_id\n")

    for i in range(len(pred_label)):
        fout_p.write(str(senti_true_label[i])+"----"+str(pred_label[i])+"----"+str(img_ids[i])+"\n")
    fout_p.close()

def inference_auxtask(model, test_dataloader, output_dir, tokenizer,logger, vis=False):

    relation_pred_list=[]
    realtion_label_list=[]
    img_id_list=[]
    relation_score_list=[]
    num_valid=0
    num_right_vg=0
    rel_acc=0
    nb_eval_examples=0
    vis_num=0
    for batch in tqdm(test_dataloader, desc="Testing_VG"):
        tokens,input_ids,input_mask, sentiment_label,\
                        img_id,img_shape,relation_label, GT_boxes,roi_boxes,img_feat,spatial_feat,box_labels=post_dataloader(batch)
            
        with torch.no_grad():

            senti_pred,ranking_loss,pred_loss,pred_score,attn_map= model(img_id=img_id,
                                                                        input_ids     = input_ids, 
                                                                        input_mask   = input_mask, 
                                                                        img_feat     = img_feat, 
                                                                        relation_label    = relation_label,
                                                                        box_labels = box_labels
                                                                        )
        current_batch_size=input_ids.size()[0]

        #-----evaluate
        ##### coarse-grained
        pred_score = pred_score.detach().cpu().numpy()  #[N*n, 100]                 #.reshape(current_batch_size,args.max_GT_boxes,-1)  # [N*n, 100]->[N, n, 100]
        relation_pred = np.argmax(pred_score, axis=1)
        tmp_rel_accuracy=np.sum(relation_pred == relation_label.cpu().numpy()) 
        rel_acc += tmp_rel_accuracy

        roi_boxes=roi_boxes.detach().cpu().numpy()  #[N, 100, 4]
        GT_boxes=GT_boxes.detach().cpu()  #[N, n,4]
        attn_map=attn_map.detach().cpu().numpy()
                
        ##### fine-grained
        for i in range(current_batch_size): #N
            if relation_label[i]!=0:                  
                num_valid+=1
                           
                ious=(torchvision.ops.box_iou(GT_boxes[i,0:1,:],torch.tensor(roi_boxes[i]))).numpy() #[1,4],[100,4]->[1,100] #如果GT是0，iou为0
                sorted_index=np.argsort(-attn_map[i])[0]
                pred_ids=sorted_index[:1]  #top K=1
                topk_max_iou=ious[0][pred_ids]
                pred_iou=topk_max_iou.max()

                if pred_iou>=0.5:
                    num_right_vg+=1
    
                if vis :
                    pred_id=attn_map[i].argmax() 
                    debug_dir = os.path.join(output_dir )
                    iid=bytes((img_id[i])).decode().split(" ")[0]  
                    try:
                        img_path ='./data/twitter_images_ori/twitter2017/'+str(iid)+".jpg"       #! modify the image path      
                    except:
                        print("Add your original image path!')
                    img = cv2.imread(img_path)
                    debug_pred(debug_dir, vis_num, input_ids[i], img, (GT_boxes[i,0:1,:]).detach().numpy(), roi_boxes[i][pred_id], pred_iou,tokenizer, senti_pred[i],sentiment_label[i],relation_pred[i],relation_label[i],pred_score[i],box_labels[i][0])
                    vis_num+=1
                                    

        relation_score_list.append(pred_score)
        relation_pred_list.append(relation_pred)
        realtion_label_list.append(relation_label.cpu().numpy())
        img_id_list.append(img_id)
        
        
                
        nb_eval_examples += current_batch_size
    
                
    rel_acc =rel_acc/ nb_eval_examples
    ranking_vg_acc=num_right_vg/num_valid

    relation_pred_label = np.concatenate(relation_pred_list)
    relation_true_label = np.concatenate(realtion_label_list)
    relation_pred_score=np.concatenate(relation_score_list)
    img_ids = np.concatenate(img_id_list)
            
    test_rel_precision, test_rel_recall, test_rel_F_score = macro_f1(relation_true_label, relation_pred_score)
            
    result = {'num_right_vg':num_right_vg,
            'num_valid':num_valid,
            'nb_eval_examples':nb_eval_examples,
            'Dev_rel_acc':rel_acc,
            'rel_precision':test_rel_precision,
            'rel_recall':test_rel_recall,
            'rel_F_score':test_rel_F_score,
            'Dev_ranking_vg_acc':ranking_vg_acc,
            }

    logger.info("***** visual grounding results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    fout_p = open(os.path.join(output_dir, "relation_pred.txt"), 'w')
    fout_p.write("ground truth ---- pred label ---- img_id\n")
    for i in range(len(relation_true_label)):
        try:
            fout_p.write(str(relation_true_label[i])+"----"+str(relation_pred_label[i])+"----"+str(img_ids[i])+"\n")
        except:
            import pdb; pdb.set_trace()  
    fout_p.close()



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default='twitter2017',
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default= './data/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--VG_data_dir",
                        default= './data/Image-Target Matching',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--imagefeat_dir",
                        default = '/mnt/nfs-storage-titan/data/twitter_images/', # default ='./data/twitter_images/',
                        type=str,
                        required=True,
                        )
    parser.add_argument("--VG_imagefeat_dir",
                        default = '/mnt/nfs-storage-titan/data/twitter_images/', # default ='./data/twitter_images/',
                        type=str,
                        required=True,
                        )
    parser.add_argument("--output_dir",
                        default="./log/",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_file",
                        default="/mnt/nfs-storage-titan/ITM/log/05-01-2022-15-39-32__twitter2017_1e-05_1e-06/pytorch_model.bin",
                        type=str,
                        required=True,
                        help="The input directory where the model has been written.")
    parser.add_argument("--vis",
                        default=True,
                        help="Whether to visualization.")


    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--roi_num', 
                        default=100, 
                        type=int)
   
    args = parser.parse_args()
    
  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    args.data_dir=args.data_dir+str(args.dataset).lower()+ '/%s.pkl'
    args.imagefeat_dir=args.imagefeat_dir+str(args.dataset).lower()
    args.VG_data_dir= args.VG_data_dir + '/%s.pkl'
    args.VG_imagefeat_dir = args.VG_imagefeat_dir+'twitter2017'

    args.output_dir=args.output_dir+str(args.dataset)+"/"
   
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_logger_file=os.path.join(args.output_dir,'log.txt')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    filename=output_logger_file)
    logger = logging.getLogger(__name__)
    
    logger.info("dataset:{} ".format(args.dataset))
    logger.info("model_dir:{}  ".format(args.model_file))
    logger.info(args)

    
    
   
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
   
    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(args.model_file)
    model = Coarse2Fine(roberta_name='roberta-base', roi_num = args.roi_num )
    model.load_state_dict(model_state_dict)
    model.to(device)
                                
    test_dataset_SA = MyDataset(args.data_dir%str('test'),args.imagefeat_dir,tokenizer,max_seq_len=args.max_seq_length,num_roi_boxes=100)
    test_dataloader_SA = Data.DataLoader(dataset=test_dataset_SA,shuffle=False, batch_size=args.eval_batch_size,num_workers=0)
    
    test_dataset_VG=MyDataset(args.VG_data_dir%str('VG_test'), args.VG_imagefeat_dir, tokenizer, max_seq_len= args.max_seq_length, num_roi_boxes=100)
    test_dataloader_VG = Data.DataLoader(dataset=test_dataset_VG,shuffle=True, batch_size=args.eval_batch_size,num_workers=0)
    
    inference_sentiment(model =model,
                        test_dataloader = test_dataloader_SA,
                        output_dir =args.output_dir,
                        logger = logger)
    
    inference_auxtask(model =model,
                        test_dataloader = test_dataloader_VG,
                        output_dir =args.output_dir,
                        tokenizer = tokenizer,
                        logger = logger,
                        vis = args.vis)

    
        
    

if __name__ == "__main__":
    main()
