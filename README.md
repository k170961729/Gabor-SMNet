Followed from https://github.com/JiaRenChang/PSMNet

## Dependencies

Python 3.5
PyTorch 0.4.0+
torchvision 0.2.0

## Dataset
KITTI 2012 / KITTI 2015
Scene Flow

KITTI: The KITTI dataset is relatively small and has real-world pictures with sparse ground truth disparity maps.
Scene Flow: The Scene Flow dataset is relatively large. It is a synthetic dataset with dense ground truth disparity maps.
Training strategy: Pretrain the network using Scene Flow dataset, then finetune the network on the KITTI dataset

## Train
python main.py --maxdisp 192 \
               --datapath /SceneFlow_dataset_path \
               --epochs 10  \
               --savemodel ./trained/

python finetune.py --maxdisp 192 \
                   --datatype 2015 \
                   --datapath /KITTI2_train_set_path \
                   --epochs 600 \
                   --loadmodel ./trained/checkpoint_10.tar \
                   --savemodel ./trained/trained2015

Evaluation:
python submission.py --loadmodel ./trained/trained2012/finetune_600.tar  \
                     --KITTI 2012
                     --datapath  /KITTI_test_set_path

## Final Result
The disparity maps for the KITTI testset are calculated by submission.py, those disparity maps are submitted to the KITTI website to calculated the final accuracy result.

## Please cite

```
@article{luan2018gabor,
  title={Gabor convolutional networks},
  author={Luan, Shangzhen and Chen, Chen and Zhang, Baochang and Han, Jungong and Liu, Jianzhuang},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4357--4366},
  year={2018},
  publisher={IEEE}
}

@inproceedings{liu2018stereo,
  title={Stereo Matching Using Gabor Convolutional Neural Network},
  author={Liu, Zhendong and Hu, Qinglei and Liu, Jiachen and Zhang, Baochang},
  booktitle={2018 11th International Workshop on Human Friendly Robotics (HFR)},
  pages={48--53},
  year={2018},
  organization={IEEE}
}
```