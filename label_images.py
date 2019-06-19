import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
import PIL
from PIL import Image, ImageDraw
from datasets import WhaleDataset, BASIC_IMAGE_T
import colorsys


def get_N_HexCol(N):
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        out.append(tuple(rgb))
    return out

labels = '''
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
street sign
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
hat
backpack
umbrella
shoe
eye glasses
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
plate
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
mirror
dining table
window
desk
toilet
door
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
blender
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
hair brush
'''.strip().split('\n')

labels = ['bground', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

@torch.no_grad()
def write_labels(dataloader, args):
    cmap = torch.tensor(get_N_HexCol(21)).to(args.device).type(torch.float) / 255.
    t = lambda i: Image.fromarray(i.clone().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to(args.device)
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=True).eval().to(args.device)
    for images, _, idx in dataloader:
        images = images.to(args.device)
        predictions = model(images)
        selected_layer = torch.argmax(predictions['out'], dim=1)
        masked_images = cmap[selected_layer].permute(0, 3, 1, 2)
        out = torch.cat([images, masked_images], dim=3)
        for i in range(images.shape[0]):
            img = t(out[i])
            '''
        for i, p in enumerate(predictions):
            img = t(images[i])
            draw = ImageDraw.Draw(img, 'RGB')
            hits = 0
            for j, l in enumerate(p['labels'].tolist()):
                x0, y0, x1, y1 = p['boxes'][j].clamp(0, 800).type(torch.int).tolist()
                if y1 <= y0 or x1 <= x0:
                    continue
                score = p['scores'][j].item()
                hits += 1
                color = cmap[l]
                draw.rectangle((x0, y0, x1, y1), outline=color)
                draw.text((x0, y0), '%s - %.2f' %(labels[l-1], score), fill=color)
                if hits >= 5:
                    break
            '''
            img.save(os.path.join(args.output_folder, '%s.jpg' % idx[i].item()))
                

def main(args):
    all_dataset = WhaleDataset(args.data_folder, image_transformation=BASIC_IMAGE_T)
    num_channels = 3

    # Define the data loaders
    loader = torch.utils.data.DataLoader(all_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Fixed images for Tensorboard
    write_labels(loader, args)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str, default='whales',
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet, whales)')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(args)
