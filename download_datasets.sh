#!/bin/bash
PATH=~/.local/bin:$PATH

pip install gdown

mkdir -p dataset_files/ori_dataset
cd dataset_files
mkdir statistics
gdown "1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0&confirm=t"
gdown "1yQ_mTwF4VzBB1_v5MB48odyXEGao2LrH&confirm=t"
gdown "1XZHXCHiA3qYRsHjF68oo5gtzU_gbGMjo&confirm=t"
gdown "1WlxXLFWpDSrCvCAIXDABSHQxIxXUdlp-&confirm=t"
gdown "1Vi3VX7tp9rClYZS3_VDNVH_yZzf5bVeg&confirm=t"
gdown "1C3-cB2YkByvjYFIOBUO0g8ornrZECkbC&confirm=t"
gdown "10xtuwzWJn_7SfOi7xnXjXTF9B3oz6zdO&confirm=t"

cd ..
unzip dataset_files/spider.zip -d dataset_files/ori_dataset/
chmod -R 777 dataset_files/
