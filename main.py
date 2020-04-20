#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader
from Data_loader import Data
import WCT
from timeit import default_timer as timer
import torchvision

#暂时不实现合成纹理synthesis，stylePair
parser = argparse.ArgumentParser(description='Universal Style Transfer via Feature Transforms',
                                epilog='Supported image file formats are: jpg, jpeg, png')
parser.add_argument('--outDir', default='outputs', help='The path to save stylized results')
parser.add_argument('--cuda', default='true', help='Use cuda or not.')
parser.add_argument('--content', default='D:/Projects/style_transfer_WCT/inputs/contents', help='The path of content.')
parser.add_argument('--style', default='D:/Projects/style_transfer_WCT/inputs/styles', help='The path of style.')
parser.add_argument('--finesize', default=400, help='resize image to fineSize x ,leave it to 0 if not resize')
parser.add_argument('--alpha', type=float, default=0.2, help='Hyperparameter balancing the blending between original content features and WCT-transformed features')

args = parser.parse_args()

def save_image(img, content_name, style_name, out_ext, args):
    torchvision.utils.save_image(img.cpu().detach().squeeze(0),
                                os.path.join(args.outDir, content_name + '-' + style_name + '-' + str(args.alpha) + '.' + out_ext)
                                )

def main():
    #生成输出文件夹
    try:
        os.makedirs(args.outDir)
    except OSError:
        pass

    #使用gpu
    if args.cuda and torch.cuda.is_available():
        print('Use cuda.')
        args.device = torch.device('cuda')
    else:
        print('Use cpu.')
        args.device = torch.device('cpu')

    dataset = Data(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = WCT.WCTmodel(args).to(device=args.device)
    model.eval()

    for i, sample in enumerate(dataloader):
        content = sample[0].to(device=args.device)
        style = sample[1].to(device=args.device)
        
        c_basename = str(os.path.basename(sample[2][0][0]).split('.')[0])
        c_ext = str(os.path.basename(sample[2][0][0]).split('.')[1])

        s_basename = str(os.path.basename(sample[2][1][0]).split('.')[0])
        s_ext = str(os.path.basename(sample[2][1][0]).split('.')[1])
        
        #计时
        start = timer()
        out = model(content, style)
        end = timer()
        print('Use time:' + str(end - start) + 's')

        save_image(out, c_basename, s_basename, c_ext, args)

if __name__=="__main__":
    main()