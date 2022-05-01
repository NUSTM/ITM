# Targeted Multimodal Sentiment Classification based on Coarse-to-Fine Grained Image-Target Matching 

Codes and datasets for our IJCAI'2022 paper:[Targeted Multimodal Sentiment Classification based on Coarse-to-Fine Grained Image-Target Matching]()

Author

Jianfei Yu & Jieming Wang

wjm@njust.edu.cn

## Data 
We adopt two kinds of datasets to systematically evaluate the effectiveness of ITM.

- Twitter datasets for the TMSC task: the processed pkl files are in floder  ./data/twitter201x/ . The original tweets, images and sentiment annotations can be download from [https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view](https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view)
- Image-Target Matching dataset for the two auxiliary tasks: the processed pkl files are in floder  ./data/Image_Target_Matching/ . The original annotated xml files can be download from [Baidu Netdist]() with code: xxxx.
  
## Image Processing 
We use [Faster-RCNN](https://github.com/peteanderson80/bottom-up-attention) to extract region feature as the input feature of images.For the details, you can refer to the [Faster-RCNN](https://github.com/peteanderson80/bottom-up-attention).
```
python ./tools/extract_feat.py --gpu 0 \
                    --cfg experiments/cfgs/faster_rcnn_end2end_resnet_vg.yml \
                    --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
                    --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel \
                    --img_dir ../ITM/data/twitter_images/twitter2017 \           
                    --out_dir ../ITM/data/twitter_images/twitter2017_extract/   
                    --num_bbox 100,100 \             
                    --feat_name pool5_flat   
```
## Code Usage
 Note that you should change data path.
- Training for ITM
```
sh train.sh
```
- Inference for ITM
```
sh test.sh
```

## Acknowledgements


