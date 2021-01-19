# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys, glob
import pickle
import cv2
import torch
import numpy as np
from time import time
import random
# from scipy.io import savemat

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

from multiprocessing import Process, set_start_method


def merge_dict(dict_list):
    dict_out = {}
    for key in dict_list[0]:
        if(isinstance(dict_list[0][key], torch.Tensor)):
            dict_out[key] = torch.cat([_dict[key] for _dict in dict_list], dim=0)
        else:
            dict_out[key] = [_dict[key] for _dict in dict_list]

    return dict_out


def expand_dict(full_dict, n_frame):
    list_of_dict = []
    for i in range(n_frame):
        _dict = {}
        for key in full_dict:
            _dict[key] = full_dict[key][i: i + 1]
        list_of_dict.append(_dict)

    return list_of_dict


def process_proc(video_list, device,
                 input_root, landmark_root,
                 savefolder,
                 iscrop=False,
                 saveMat=True, saveVis=False, saveImages=True):

    deca_cfg.model.use_tex = True
    deca = DECA(config=deca_cfg, device=device)
    testdata = datasets.VideoTestData(video_list, input_root, landmark_root, iscrop=iscrop)
    dataloader = torch.utils.data.DataLoader(testdata, num_workers=8)

    print(f'{len(video_list)} items.')
    batchsize = 64
    for batch in dataloader:
        name = batch['videoname'][0]
        print(name)
        if(name == 'None'):
            print('Skipped.')
            continue

        name, _ = os.path.splitext(name)
        out_name = name.replace(input_root, savefolder)
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        out_coeff_name = out_name + '.npz'

        if(os.path.exists(out_coeff_name)):
            print('Skipped.')
            continue

        videos = batch['video'][0].to(device)
        n_frame = videos.shape[0]
        fid = 0
        visdict_list = []
        codedict_list = []
        for k in range(0, n_frame, batchsize):
            video_in = videos[k: k + batchsize]
            codedict = deca.encode(video_in)
            codedict_list.append(codedict)
            if(saveVis is True or saveImages is True and k == 0):
                vis_code_dict = dict()
                for key in codedict:
                    vis_code_dict[key] = codedict[key][0:1]
                _, visdict = deca.decode(vis_code_dict)  # tensor
                visdict_list.append(visdict)

        codedict_final = merge_dict(codedict_list)

        if saveImages or saveVis:
            visdict_final = merge_dict(visdict_list)
            visdict_expand = expand_dict(visdict_final, n_frame)
            os.makedirs(out_name, exist_ok=True)
        # -- save results
        if saveMat:
            npy_dict = util.dict_tensor2npy(codedict_final, ignore_key_list=['images'], batch_mode=True)
            if(iscrop):
                npy_dict['tforms'] = batch['tform_video'][0].cpu().numpy()
            np.savez(out_coeff_name, npy_dict)
        if saveVis:
            cv2.imwrite(os.path.join(out_name, 'vis{:05d}.jpg'.format(fid)), deca.visualize(visdict_expand[0]))
        if saveImages:
#             ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'normal_images']:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'normal_images', 'uv_images', 'uv_textures']:
                if vis_name not in visdict_expand[0].keys():
                    continue
                image = util.tensor2image(visdict_expand[0][vis_name][0])
                cv2.imwrite(os.path.join(out_name, vis_name + '{:05d}.jpg'.format(fid)), image)
    print(f'-- please check the results in {savefolder}')


if __name__ == "__main__":
    input_root = '/mnt/data/voxceleb/dev/mp4'
    landmark_root = '/mnt/data/voxceleb/dev/landmark_2d'
    savefolder = '/mnt/data/voxceleb/flame'

    iscrop = True
    gpu_device = 'cuda'

    video_pkl = sys.argv[1]
    with open(video_pkl, 'rb') as f:
        video_list = pickle.load(f)
    # video_list = glob.glob(input_root + '/**/*.mp4', recursive=True)

    os.makedirs(savefolder, exist_ok=True)

    process_proc(video_list,
                 gpu_device, input_root, landmark_root, savefolder,
                 iscrop,
                 True, True, False)
