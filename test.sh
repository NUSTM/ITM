for i in 'twitter2017' # 'twitter2017'
do 
    python test.py 
    --dataset ${i} \
    --data_dir ./data/ \
    --VG_data_dir ./data/Image-Target Matching \
    --imagefeat_dir /mnt/nfs-storage-titan/data/twitter_images/ \
    --VG_imagefeat_dir /mnt/nfs-storage-titan/data/twitter_images/ \
    --output_dir ./log/ \
    --model_file ./${i}/pytorch_model.bin \
    --vis 
done


