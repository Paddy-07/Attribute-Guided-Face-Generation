"""Evaluate the discriminator and generator on held-out CelebA images.

Reports real/fake discrimination accuracy and attribute classification
accuracy on validation images, then checks the discriminator's attribute
predictions for generated faces. Generated-face attribute scores are a
useful diagnostic, but they are not an independent measure of image quality.

Usage:
    python evaluate.py --checkpoint ./checkpoints/ckpt_epoch050.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from config import config
from data.dataset import CelebADataset
from models.discriminator import Discriminator
from models.generator import Generator


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint containing G and D state dictionaries")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=0,
                        help="optional limit for a quick check; 0 evaluates all validation data")
    parser.add_argument("--workers", type=int, default=config.num_workers)
    return parser.parse_args()


def main():
    args = get_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")

    device = config.device
    print(f"Device: {device}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    G = Generator(config.latent_dim, config.num_attrs,
                  config.g_conv_dim, config.channels).to(device)
    D = Discriminator(config.num_attrs,
                      config.d_conv_dim, config.channels).to(device)
    G.load_state_dict(checkpoint["G"])
    D.load_state_dict(checkpoint["D"])
    G.eval()
    D.eval()

    dataset = CelebADataset(
        img_dir=config.img_dir,
        attr_path=config.attr_path,
        selected_attrs=config.selected_attrs,
        image_size=config.image_size,
        split="val",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Validation examples: {len(dataset)}")

    # Binary discriminator confusion counts: positive means predicted real.
    real_tp = real_fn = fake_tn = fake_fp = 0
    real_attr_correct = torch.zeros(config.num_attrs, dtype=torch.long)
    fake_attr_correct = torch.zeros(config.num_attrs, dtype=torch.long)
    total_attr_labels = 0
    total_images = 0

    with torch.inference_mode():
        for batch_index, (real_images, real_attrs) in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break

            real_images = real_images.to(device, non_blocking=True)
            real_attrs = real_attrs.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            real_logit, real_attr_logit = D(real_images)
            z = torch.randn(batch_size, config.latent_dim, device=device)
            # Use real label combinations so generated conditions follow the
            # attribute frequencies and correlations in CelebA.
            fake_attrs = real_attrs
            fake_images = G(z, fake_attrs)
            fake_logit, fake_attr_logit = D(fake_images)

            real_pred = real_logit.view(-1) >= 0
            fake_pred = fake_logit.view(-1) >= 0
            real_tp += int(real_pred.sum().item())
            real_fn += batch_size - int(real_pred.sum().item())
            fake_fp += int(fake_pred.sum().item())
            fake_tn += batch_size - int(fake_pred.sum().item())

            real_attr_pred = real_attr_logit >= 0
            fake_attr_pred = fake_attr_logit >= 0
            real_attr_correct += (real_attr_pred == (real_attrs >= 0.5)).sum(dim=0).cpu()
            fake_attr_correct += (fake_attr_pred == (fake_attrs >= 0.5)).sum(dim=0).cpu()
            total_attr_labels += batch_size
            total_images += batch_size

    if total_images == 0:
        raise RuntimeError("No validation batches were evaluated")

    real_accuracy = real_tp / (real_tp + real_fn)
    fake_accuracy = fake_tn / (fake_tn + fake_fp)
    balanced_accuracy = (real_accuracy + fake_accuracy) / 2
    real_attr_accuracy = real_attr_correct.sum().item() / (total_attr_labels * config.num_attrs)
    fake_attr_accuracy = fake_attr_correct.sum().item() / (total_attr_labels * config.num_attrs)

    print("\nDiscriminator real/fake results (threshold = 0.5):")
    print(f"  Real validation faces correctly called real: {real_accuracy:.3%}")
    print(f"  Generated faces correctly called fake:       {fake_accuracy:.3%}")
    print(f"  Balanced real/fake accuracy:                 {balanced_accuracy:.3%}")
    print("\nDiscriminator attribute accuracy:")
    print(f"  On real validation faces: {real_attr_accuracy:.3%}")
    print(f"  On generated faces vs requested labels: {fake_attr_accuracy:.3%}")
    print("\nPer-attribute accuracy (real / generated):")
    for index, name in enumerate(config.selected_attrs):
        real_score = real_attr_correct[index].item() / total_attr_labels
        fake_score = fake_attr_correct[index].item() / total_attr_labels
        print(f"  {name:16s} {real_score:7.2%} / {fake_score:7.2%}")
    print(f"\nEvaluated {total_images} real validation faces and the same number of generated faces.")
    print("Note: generated-face attribute accuracy comes from this discriminator's own classifier;")
    print("it is a diagnostic, not an independent proof that the requested traits look correct.")


if __name__ == "__main__":
    main()
