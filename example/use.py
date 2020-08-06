import argparse

import torch
import torch.nn.parallel
import datasets
from utils import AverageMeter, img_cvt, img_silhouettes
import soft_renderer as sr
import soft_renderer.functional as srf
import models
import time
import os
import imageio
import numpy as np



from losses import singleview_iou_loss, multiview_iou_loss

BATCH_SIZE = 2
IMAGE_SIZE = 64
# CLASS_IDS_ALL = (
#     '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
#     '03691459,04090263,04256520,04379243,04401088,04530566')
CLASS_IDS_ALL = (
    '02691156')

PRINT_FREQ = 100
SAVE_FREQ = 100

MODEL_DIRECTORY = './data/models'
MODEL_DIRECTORY = '/home/desperado/Documents/3DReconstruction/SoftRas/data/results/models/recon/checkpoint_0210000.pth.tar'

DATASET_DIRECTORY = './data/datasets'

SIGMA_VAL = 0.01
IMAGE_PATH = ''

LEARNING_RATE = 1.0
LR_TYPE = 'step'

NUM_ITERATIONS = 250000

LAMBDA_LAPLACIAN = 5e-3
LAMBDA_FLATTEN = 5e-4
EXPERIMENT_ID = 'recon'
# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str, default=EXPERIMENT_ID)
parser.add_argument('-d', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)

parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('-img', '--image-path', type=str, default=IMAGE_PATH)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

parser.add_argument('-lr', '--learning-rate', type=float, default=LEARNING_RATE)
parser.add_argument('-lrt', '--lr-type', type=str, default=LR_TYPE)

parser.add_argument('-ll', '--lambda-laplacian', type=float, default=LAMBDA_LAPLACIAN)
parser.add_argument('-lf', '--lambda-flatten', type=float, default=LAMBDA_FLATTEN)
parser.add_argument('-ni', '--num-iterations', type=int, default=NUM_ITERATIONS)

parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
args = parser.parse_args()

device = torch.device("cuda:0")

# setup model & optimizer
model = models.Model('data/obj/sphere/sphere_642.obj', args=args)
model = model.cuda()

state_dicts = torch.load(args.model_directory)
model.load_state_dict(state_dicts['model'], strict=False)
model.eval()

dataset_val = datasets.ShapeNet(args.dataset_directory, args.class_ids.split(','), 'train')

directory_output = './data/results/test'
os.makedirs(directory_output, exist_ok=True)
directory_mesh = os.path.join(directory_output, args.experiment_id)
os.makedirs(directory_mesh, exist_ok=True)

directory_mesh_cls = os.path.join(directory_mesh, 'MESH')
os.makedirs(directory_mesh_cls, exist_ok=True)

# load images from multi-view
images_a, images_b, viewpoints_a, viewpoints_b = dataset_val.get_random_same_class_batch(args.batch_size)
images_a = images_a.cuda()
images_b = images_b.cuda()
viewpoints_a = viewpoints_a.cuda()
viewpoints_b = viewpoints_b.cuda()


# def adjust_learning_rate(optimizers, learning_rate, i, method):
#     if method == 'step' and lr > 0.1:
#         lr, decay = learning_rate, 0.3
#         if i >= 1000:
#             lr *= decay
#     elif method == 'constant':
#         lr = learning_rate
#     else:
#         print("no such learing rate type")

#     for optimizer in optimizers:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#     return lr

def test():
    vertices, faces, latentvector, silhouettes = model(images=images_a, viewpoints=viewpoints_a, task='use_test')
    print('latentvector_GT', latentvector.sum())
    for k in range(vertices.size(0)):
        obj_id = k
        mesh_path = os.path.join(directory_mesh_cls, 'X%06d.obj' % obj_id)
        ref_path = os.path.join(directory_mesh_cls, 'X%06d_ref.png' % obj_id)
        render_path = os.path.join(directory_mesh_cls, 'X%06d_render.png' % obj_id)
        srf.save_obj(mesh_path, vertices[k], faces[k])
        imageio.imsave(render_path, img_cvt(silhouettes[k]))
        imageio.imsave(ref_path, img_cvt(images_a[k]))
    
    return latentvector.clone().detach().cpu().numpy()

def use():
    learning_rate = 0.1

    ## !!!!! (batch_size, 1, 512) before is wrong
    latentvector = torch.randn(args.batch_size,512,device=device,requires_grad=True)
    # latentvector = torch.from_numpy(test())
    # latentvector = latentvector + torch.randn_like(latentvector)
    # latentvector.requires_grad = True
    # latentvector = latentvector.cuda()

    print(latentvector)
    print('--',latentvector.size(0))

    optimizer = torch.optim.Adam([latentvector], lr=1e-3)
    plot_period = 1000
    # latentvector = latentvector.cuda()

    for i in range(args.num_iterations):
        # soft render images
        # silhouettes vertices faces are all tuple
        silhouettes, laplacian_loss, flatten_loss, vertices, faces = model(latentvector=latentvector, 
                                                        viewpoints=[viewpoints_a, viewpoints_b],
                                                        task='use')
        vertices = vertices[0]
        faces = faces[0]
        if i == 0:
            print('===Train===')
            print('silhouettes', silhouettes[0].size())
            print('images_a', images_a.size())
            print('vertices', vertices.shape)
            print('faces', faces.shape)

        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()

        # compute loss
        loss = multiview_iou_loss(silhouettes, images_a, images_b)
            #    args.lambda_laplacian * laplacian_loss + \
            #    args.lambda_flatten * flatten_loss + \
            #    latentvector.pow(2).sum()
        print('loss', loss, i)

        # loss = singleview_iou_loss(silhouettes, images_a) + \
        #         args.lambda_laplacian * laplacian_loss.mean()

        # loss = torch.sum((silhouettes[:,3,...] - images_a[:,3,...]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # because silhouettes is a tuple
        silhouettes_a = silhouettes[0]
        silhouettes_b = silhouettes[1]
        if i % plot_period == 0:
            if learning_rate > 0.1:
                learning_rate = learning_rate * 0.8
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            print('=================SAVE================', i)
            for k in range(vertices.size(0)):
                obj_id = i+k
                mesh_path = os.path.join(directory_mesh_cls, '%06d.obj' % obj_id)
                render_path = os.path.join(directory_mesh_cls, '%06d_render_sil.png' % obj_id)
                ref_path = os.path.join(directory_mesh_cls, '%06d_ref_rgba.png' % obj_id)
                ref_silhouettes_path = os.path.join(directory_mesh_cls, '%06d_ref_sil.png' % obj_id)

                srf.save_obj(mesh_path, vertices[k], faces[k])
                imageio.imsave(render_path, img_silhouettes(silhouettes_a[k]))
                imageio.imsave(ref_path, img_cvt(images_a[k]))
                imageio.imsave(ref_silhouettes_path, img_silhouettes(images_a[k]))

            print('lr', learning_rate)
            print('loss', loss)
            print('latentvector', latentvector.sum())
    


    print('=================================')


use()