from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from modules import VectorQuantizedVAE, to_scalar, ResBlock
from dataset import WhaleDataset, BASIC_IMAGE_T

from tensorboardX import SummaryWriter


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# desired depth layers to compute style/content losses :
content_layers_default = ['res_3', 'res_4']
style_layers_default = ['conv_1', 'conv_2', 'res_3', 'res_4']

def get_style_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential()

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        elif isinstance(layer, ResBlock):
            i += 1
            name = 'res_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    def compute_loss():
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss
        return style_score + content_score

    return model, compute_loss


class CoordinateGenerator(nn.Module):
    def __init__(self, input_dim):
        super(CoordinateGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 8, 2, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 4, 2, 1, 1),
        )
        self._index_range = torch.arange(0, 16)

    def forward(self, x):
        # (-1, dim, 16, 16)
        coords_vote = self.encoder(x)
        assert coords_vote.shape == [x.shape[0], 4, 16, 16]
        _p = lambda v, dim: F.softmax(v.sum(dim=dim), dim=2)
        x_votes = _p(coords_vote, dim=2)
        y_votes = _p(coords_vote, dim=3)
        x_coord = x_votes * self._index_range / 16
        y_coord = y_votes * self._index_range / 16
        return torch.cat([x_coord, y_coord], dim=1)


class AffineCropper(nn.Module):
    def __init__(self, dataloader, encoder):
        super(AffineCropper, self).__init__()
        self.dataloader = dataloader
        self.encoder = encoder
        self.generator = CoordinateGenerator(input_dim)
        self.h, self.w = 128, 128
        self.points_dest = torch.FloatTensor([[
            [0, 0], [self.w - 1, 0], [self.w - 1, self.h - 1], [0, self.h - 1],
        ]])

    def sample(self, idx, points_source):
        image_id = self.dataloader._all_records.iloc[idx]['Image']
        img = self.dataloader.read_image(image_id)
        points_source[1] *= img.shape[1]
        points_source[2] *= img.shape[2]
        M = kornia.get_perspective_transform(points_source, self.points_dest)
        return kornia.warp_perspective(img, M, dsize=(self.h, self.w))

    def forward(self, x, idxs):
        with torch.no_grad():
            latent = self.encoder(x)
        aspect_coords = self.generator(latent).clamp(0, 1)

        cropped_images = []
        for i in range(idx.shape[0]):
            idx = idxs[i]
            img_result = affine_resample.sample(idx, aspect_coords[i])
            cropped_images.append(img_result)
        cropped_images = torch.cat(cropped_images, dim=0)
        return cropped_images


def train(data_loader, model, affine_resample, optimizer, args, writer):
    style_img = content_img = data_loader.process_image(args.target_image)
    style_transfer_model, get_loss = get_style_model_and_losses(model,
        style_img, content_img)
    style_transfer_model.to(args.device)
    for images, _, idx in data_loader:
        images = images.to(args.device)
        idx = idx.to(args.device)

        optimizer.zero_grad()
        cropped_images = affine_resample(images, idx)
        style_transfer_model(cropped_images)

        loss = get_loss()
        loss.backward()

        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)

        optimizer.step()
        args.steps += 1


def test(data_loader, model, affine_resample, args, writer):
    style_img = content_img = data_loader.process_image(args.target_image)
    style_transfer_model, get_loss = get_style_model_and_losses(model,
        style_img, content_img)
    with torch.no_grad():
        loss = 0.
        for images, _, idx in data_loader:
            images = images.to(args.device)
            idx = idx.to(args.device)
            cropped_images = affine_resample(images, idx)
            style_transfer_model(cropped_images)

            loss += get_loss()
        loss  /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test', loss.item(), args.steps)

    return loss.item()

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde = model(images)
    return x_tilde


def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = './models/{0}'.format(args.output_folder)

    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif args.dataset == 'whales':
        # Define the train, valid & test datasets
        all_dataset = WhaleDataset(args.data_folder, image_transformation=BASIC_IMAGE_T)
        from torch.utils.data import random_split
        k = len(all_dataset)
        train_s = int(k * .6)
        test_s = int(k * .2)
        valid_s = k - train_s - test_s
        train_dataset, test_dataset, valid_dataset = random_split(all_dataset, (train_s, test_s, valid_s))
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    #TODO model = torch.load(args.model_path).encoder
    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).encoder.to(args.device)
    affine_resample = AffineCropper(args.data_folder, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction', grid, 0)

    best_loss = -1.
    for epoch in range(args.num_epochs):
        train(train_loader, model, affine_resample, optimizer, args, writer)
        loss = test(valid_loader, model, affine_resample, args, writer)

        reconstruction = generate_samples(fixed_images, affine_resample, args)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('cropped', grid, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Aspect Transfer')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str, default='whales',
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--model-path', type=str,
        help='path to trained autoencoder')
    parser.add_argument('--target-image', type=str,
        help='path to target image')

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
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
