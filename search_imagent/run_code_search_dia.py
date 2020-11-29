import os
import numpy as np
import random

for seed in ['666']:
    for num_train in ['450']:
        for num_val in ['50']:
            for data_aug in ['1']:
                    name = 'ean_search'
                    os.system('CUDA_VISIBLE_DEVICES=6,7 '+
                              'python EAN_search.py '+
                              ' --batch-size 1000 '+
                              '-a forward_dia_fbresnet50 '+
                              ' --manualSeed ' + seed +
                              ' -data /home/jovyan/imagenet/ILSVRC2012_Data --checkpoint ckpts/'+ name +
                              ' --validate_size 2'+
                              ' --resume /home/jovyan/NAS_ckpts/ensemble_dia_train_on_subset/forward_dia_fbresnet50/checkpoint39.pth.tar')