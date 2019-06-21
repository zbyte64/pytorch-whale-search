import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import kornia
import copy
import math

from modules import VectorQuantizedVAE, to_scalar, ResBlock, Lambda
from datasets import (
    TripletWhaleDataset,
    WhaleDataset,
    BASIC_IMAGE_T,
    FINALIZE_T,
    AUG_IMAGE_T,
)

from tensorboardX import SummaryWriter


class MassBlock(nn.Module):
    """
    Computes center of mass for each channel, and appends normalized mass
    returns [BxCx3]
    """

    def __init__(self, w, h):
        super().__init__()
        xe = torch.arange(0, 1, 1 / w)
        ye = torch.arange(0, 1, 1 / h)
        xv, yv = torch.meshgrid(xe, ye)
        self.coords = torch.stack([xv, yv])

    def forward(self, x):
        mx = x.sum((2, 3))  # [b, c]
        cm = torch.einsum("bcij,dij->bdc", x, self.coords) / mx
        cm = cm.transpose(0, 2, 1)  # [b, c, 2]
        nm = torch.softmax(mx, dim=1).unsqueeze(2)
        return torch.cat([cm, nm], dim=2)


class EmbedSpace(nn.Module):
    """
    Conduct an identity convolution after appending channels that indicate tensor location
    """

    def __init__(self, dim, w, h):
        super().__init__()
        xe = torch.arange(0, w, dtype=torch.float32) / w * math.pi
        ye = torch.arange(0, h, dtype=torch.float32) / h * math.pi
        xec = torch.cos(xe)
        xes = torch.sin(xe)
        yec = torch.cos(ye)
        yes = torch.sin(ye)
        self.space_channels = torch.stack(
            [*torch.meshgrid(xes, yes), *torch.meshgrid(xec, yec)], dim=0
        )
        self.conv = nn.Conv2d(dim + 4, dim, 1, 1)

    def forward(self, x):
        x = torch.cat([x, self.space_channels.repeat(x.shape[0], 1, 1, 1)], dim=1)
        x = self.conv(x)
        return x


class AffineClassifier(nn.Module):
    def __init__(self, dataloader, encoder, output_dim):
        super().__init__()
        self.dataloader = dataloader
        self._transfer_ = {"encoder": encoder}
        self.region_proposals = nn.Sequential(
            nn.Conv2d(256, 64, 4, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 7, 1, 1),
            ResBlock(16),
            ResBlock(16),
            MassBlock(32, 32),
            Lambda(lambda x: x.view(x.shape[0], 16 * 3)),
            nn.Linear(16 * 3, 4),
            nn.Sigmoid(),
        )
        self.h, self.w = 128, 128
        self.points_dest = torch.FloatTensor(
            [[[0, 0], [self.w - 1, 0], [self.w - 1, self.h - 1], [0, self.h - 1]]]
        )
        self.conv_classifier = nn.Sequential(
            nn.Conv2d(256, 128, 4, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 7, 1, 1),
            ResBlock(64),
            ResBlock(64),
            EmbedSpace(64, 32, 32),
            nn.MaxPool2d(32),
        )
        self.fn = nn.Sequential(nn.Linear(64, output_dim))

    def forward(self, x, idx):
        px = self._transfer_["encoder"](x)
        rp = self.region_proposals(px).view(x.shape[0], -1)
        py = self.conv_classifier(px).view(x.shape[0], -1)
        p = self.fn(py)

        resamples = []
        for i in range(x.shape[0]):
            rt = self.resample_record(idx[i].item(), rp[i])
            resamples.append(rt)
        resamples = torch.stack(resamples, dim=0)
        rx = self._transfer_["encoder"](x)
        ry = self.conv_classifier(rx).view(x.shape[0], -1)
        r = self.fn(ry)

        c = p + r
        l = py + ry
        return c, l, resamples

    def resample_record(self, idx, p):
        from PIL import ImageOps

        r = self.dataloader._all_records.iloc[idx]
        image_id = r["Image"]
        img = self.dataloader.read_image(image_id, r["flipped"])
        _xy = min(img.size)
        img = img.resize((_xy, _xy))
        img = AUG_IMAGE_T(image=np.array(img))["image"]
        img = FINALIZE_T(image=img)["image"].to(p.device)
        ih, iw = img.shape[1:]
        hiw, hih = iw // 2, ih // 2
        hsw, hsh = self.w // 2, self.h // 2
        rclamp = lambda x, _min, _max: x * (_max - _min) + _min
        w = rclamp(p[0], hsw, hiw)
        h = rclamp(p[1], hsh, hih)
        cx = rclamp(p[2], hsw, iw - hsw)
        cy = rclamp(p[3], hsh, ih - hsh)
        x1 = (cx - w).clamp(0, iw)
        x2 = (cx + w).clamp(0, iw)
        y1 = (cy - h).clamp(0, ih)
        y2 = (cy + h).clamp(0, ih)
        # compute perspective transform
        points_src = torch.stack(
            [
                torch.stack([y1, x1]),
                torch.stack([y1, x2]),
                torch.stack([y2, x1]),
                torch.stack([y2, x2]),
            ]
        ).unsqueeze(0)
        img_warp = kornia.crop_and_resize(
            img.unsqueeze(0), points_src, (self.h, self.w)
        )
        # M = kornia.get_perspective_transform(points_src, self.points_dest.to(p.device))

        # warp the original image by the found transform
        # img_warp = kornia.warp_perspective(img.unsqueeze(0), M, dsize=(self.h, self.w))
        return img_warp.squeeze(0)


def train(data_loader, model, optimizer, args, writer):
    for batch in data_loader:
        (images, labels, idx, pimages, plabels, pidx, nimages, nlabels, nidx) = map(
            lambda a: a.to(args.device), batch
        )

        optimizer.zero_grad()
        p_labels, a, rsample = model(images, idx)
        _, b, _ = model(pimages, pidx)
        _, c, _ = model(nimages, nidx)
        loss_margin = F.triplet_margin_loss(a, b, c, margin=0.3, swap=True)
        # loss_class = F.cross_entropy(p_labels, labels)
        bs = labels.shape[0]
        k = p_labels.shape[1]
        loss_class = (
            kornia.losses.focal_loss(
                p_labels.view(bs, k, 1, 1).cpu(), labels.view(bs, 1, 1).cpu(), 0.5
            )
            .to(args.device)
            .mean()
        )
        loss = loss_margin * 10 + loss_class
        loss.backward()

        # Logs
        writer.add_scalar("loss/train", loss.item(), args.steps)

        optimizer.step()
        args.steps += 1


def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss = 0.0
        correct = 0.0
        count = 0
        for (
            images,
            labels,
            idx,
            pimages,
            plabels,
            pidx,
            nimages,
            nlabels,
            nidx,
        ) in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            idx = idx.to(args.device)
            p_labels, _, rimages = model(images, idx)

            loss += F.cross_entropy(p_labels, labels)
            correct += torch.sum(p_labels.argmax() == labels)
            count += images.shape[0]
        loss /= len(data_loader)
        acc = correct * 100 / count

    # Logs
    writer.add_scalar("accuracy/test", acc.item(), args.steps)
    writer.add_scalar("loss/test", loss.item(), args.steps)
    grid = make_grid(rimages.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image("resample", grid, args.steps)

    return loss.item()


def main(args):
    writer = SummaryWriter("./logs/{0}".format(args.output_folder))
    save_filename = "./models/{0}".format(args.output_folder)

    # Define the train, valid & test datasets
    all_dataset = TripletWhaleDataset(args.data_folder, min_instance_count=10)
    from torch.utils.data import random_split

    k = len(all_dataset)
    train_s = int(k * 0.6)
    test_s = int(k * 0.2)
    valid_s = k - train_s - test_s
    train_dataset, test_dataset, valid_dataset = random_split(
        all_dataset, (train_s, test_s, valid_s)
    )
    num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _, fixed_ids = next(iter(test_loader))[:3]
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image("original", fixed_grid, 0)
    n_classes = len(all_dataset.index_to_class)

    vae = VectorQuantizedVAE(num_channels, args.hidden_size, args.k)
    vae.load_state_dict(torch.load(args.model_path))
    encoder = vae.encoder.eval().to(args.device)
    model = AffineClassifier(all_dataset, encoder, n_classes).to(args.device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = -1.0
    for epoch in range(args.num_epochs):
        with torch.autograd.set_detect_anomaly(True):
            train(train_loader, model, optimizer, args, writer)
        loss = test(valid_loader, model, args, writer)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open("{0}/best.pt".format(save_filename), "wb") as f:
                torch.save(model.state_dict(), f)
        with open("{0}/model_{1}.pt".format(save_filename, epoch + 1), "wb") as f:
            torch.save(model.state_dict(), f)


if __name__ == "__main__":
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Classifier")

    # General
    parser.add_argument("--data-folder", type=str, help="name of the data folder")
    parser.add_argument(
        "--dataset",
        type=str,
        default="whales",
        help="name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)",
    )
    parser.add_argument("--model-path", type=str, help="path to trained autoencoder")

    # Latent space
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="size of the latent vectors (default: 256)",
    )
    parser.add_argument(
        "--k", type=int, default=512, help="number of latent vectors (default: 512)"
    )

    # Optimization
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size (default: 128)"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=100, help="number of epochs (default: 100)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="learning rate for Adam optimizer (default: 2e-4)",
    )

    # Miscellaneous
    parser.add_argument(
        "--output-folder",
        type=str,
        default="classifier",
        help="name of the output folder (default: classifier)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=mp.cpu_count() - 1,
        help="number of workers for trajectories sampling (default: {0})".format(
            mp.cpu_count() - 1
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="set the device (cpu or cuda, default: cpu)",
    )

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Slurm
    if "SLURM_JOB_ID" in os.environ:
        args.output_folder += "-{0}".format(os.environ["SLURM_JOB_ID"])
    if not os.path.exists("./models/{0}".format(args.output_folder)):
        os.makedirs("./models/{0}".format(args.output_folder))
    args.steps = 0

    main(args)
