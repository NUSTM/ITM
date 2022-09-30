# Targeted Multimodal Sentiment Classification based on Coarse-to-Fine Grained Image-Target Matching 

Codes and datasets for our IJCAI'2022 paper: [Targeted Multimodal Sentiment Classification based on Coarse-to-Fine Grained Image-Target Matching](https://www.ijcai.org/proceedings/2022/0622.pdf)

Author

Jianfei Yu & Jieming Wang

wjm@njust.edu.cn

## Data 
We adopt two kinds of datasets to systematically evaluate the effectiveness of ITM.

- Twitter datasets for the TMSC task: the processed pkl files are in floder  `./data/Sentiment_Analysis/twitter201x/` . The original tweets, images and sentiment annotations can be download from [https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view](https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view)
- Image-Target Matching dataset for the two auxiliary tasks: the processed pkl files are in floder  `./data/Image_Target_Matching/` . The original annotated xml files can be download from [Baidu Netdist](https://pan.baidu.com/s/1S5R9Joo5d5-kBx2L8lAPBA) with code: `rm6j`. Images of ITM are from twitter2017 dataset.

* pkl files format 
```
{'1': {'iid': '16_05_01_105', 
      'sentence': 'Safety $T$ has signed a free agent contract with the Cleveland Browns ! # BBN # NFLCats # WeAreUKStill', 
      'aspect': 'A . J . Stamps', 
      'sentiment': '2',    ## for positive
      'relation': '1',     ## for related
      'boxes': [(182, 9, 853, 856)]  
      },
 ...
```


## Image Processing 
We use [Faster-RCNN](https://github.com/peteanderson80/bottom-up-attention) to extract region feature as the input feature of images.For the details, you can refer to the original Github. Our processed image feature can be download from [Baidu Netdist](https://pan.baidu.com/s/17e6TySS5ISaITps_vf3F8w ) with code `fv25` or [GoogleDrive](https://drive.google.com/drive/folders/1So6nPbXaBnblg_oJkWACVrgXo1Epn1vZ?usp=sharing).
```
python ./tools/extract_feat.py --gpu 0 \
                    --cfg experiments/cfgs/faster_rcnn_end2end_resnet_vg.yml \
                    --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
                    --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel \
                    --img_dir ../ITM/data/twitter_images/twitter2017_ori/ \           
                    --out_dir ../ITM/data/twitter_images/twitter2017/ \   
                    --num_bbox 100,100 \             
                    --feat_name pool5_flat   
```
## Code Usage
 Note that you should change data path.
- Training for ITM
```
 python train.py 
    --dataset ${i} \
    --data_dir ./data/Sentiment_Analysis/ \
    --VG_data_dir ./data/Image-Target Matching/ \
    --imagefeat_dir ./data/twitter_images/ \
    --VG_imagefeat_dir ./data/twitter_images/ \
    --output_dir ./log/ 
```
- Inference for ITM
```
python test.py 
    --dataset ${i} \
    --data_dir ./data/Sentiment_Analysis/ \
    --VG_data_dir ./data/Image-Target Matching/ \
    --imagefeat_dir ./data/twitter_images/ \
    --VG_imagefeat_dir ./data/twitter_images/ \
    --output_dir ./log/ \
    --model_file pytorch_model.bin 
```

## Acknowledgements

- Using the Image-Target Matching dataset means you have read and accepted the copyrights set by Twitter and dataset providers.
- Most of the codes are based on the codes provided by [huggingface](https://github.com/huggingface/transformers).
