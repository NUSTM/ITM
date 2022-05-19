for i in 'twitter2015' 'twitter2017'
do 
    echo ${i}
    python train.py --dataset ${i} \
    --data_dir ./data/ \
    --VG_data_dir ./data/Image_Target_Matching/ \
    --imagefeat_dir /mnt/nfs-storage-titan/data/twitter_images/ \
    --VG_imagefeat_dir /mnt/nfs-storage-titan/data/twitter_images/ \
    --output_dir ./log/ \ 
done
