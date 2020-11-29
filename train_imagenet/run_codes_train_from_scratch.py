import os
import numpy as np
import random

for seed in ['1']:
    for _ in range(1):
        config_limit = (3, 4, 6, 3)
        seg = np.cumsum(config_limit)

        ######################### connection scheme #####################################
        # share sge
        binary_code = '1001011010011011'
        # DIA
        # binary_code = '1100011110010011'

        assert len(binary_code) == sum(config_limit), 'error in the length'
        print(len(binary_code), binary_code.count('0'))
        code1 = int(binary_code[0:seg[0]], 2)
        code2 = int(binary_code[seg[0]:seg[1]], 2)
        code3 = int(binary_code[seg[1]:seg[2]], 2)
        code4 = int(binary_code[seg[2]:seg[3]], 2)
        config = (code1, code2, code3, code4)
        print('connection scheme: ', binary_code[0:seg[0]], binary_code[seg[0]:seg[1]], binary_code[seg[1]:seg[2]], binary_code[seg[2]:seg[3]], ' Decimal: ', config)
        code_str = ''
        code_str_ = ''
        for idx, code_value in enumerate(config):
            code_str = code_str + ' ' + str(code_value)
            code_str_ = code_str_ + '_' + str(code_value)

        name = 'dia_resnet50_config'+code_str_+'_seed_'+seed

        command = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet/train_imagenet_forward.py -a forward_config_share_sge_resnet50 -data /home/jovyan/ILSVRC2012_Data --checkpoint NAS_ckpts/'+name+' --config_codes '+code_str
        # command = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet/train_imagenet_forward.py -a forward_dia_fbresnet50 -data /home/jovyan/ILSVRC2012_Data --checkpoint NAS_ckpts/' + name + ' --config_codes ' + code_str

        os.system(command)