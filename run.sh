#!/bin/bash

#python main.py --maxdisp 192 \
 #              --datapath /home/zhendong/dataset/SceneFlow/ \
  #             --epochs 10 \
   #            --savemodel ./trained/



#python finetune.py --maxdisp 192 \
 #                  --datatype 2012 \
  #                 --datapath /home/lzd/dataset/KITTI/2012/training/ \
   #                --epochs 800 \
    #               --loadmodel ./trained/checkpoint_10.tar \
     #              --savemodel ./trained/trained2012_full/

python finetune.py --maxdisp 192 \
                   --datatype 2015 \
                   --datapath /home/lzd/dataset/KITTI/2015/training/   \
                   --epochs 600 \
                   --loadmodel ./trained/trained2015/finetune_533.tar  \
                   --savemodel ./trained/trained2015/

