"""
Generate faces from a trained checkpoint with specific requested attributes.

Usage:
    python generate.py --checkpoint ./checkpoints/ckpt_epoch050.pt \
        --attrs "Male,Smiling,Eyeglasses" --num_samples 8 --out generated.png
"""

import argparse
from PIL import Image, ImageEnhance, ImageOps

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from config import config
from data.dataset import _read_attr_file
from models.discriminator import Discriminator
from models.generator import Generator


def parse_attr_string(attr_string, selected_attrs, num_samples, attr_path,
                      strict=False):
    """Build conditions for requested traits, sampling natural CelebA labels
    for traits the user did not specify unless strict mode is requested."""
    tokens = [a.strip() for a in attr_string.split(",") if a.strip()] if attr_string else []
    requested = {}
    unknown = []
    for name in tokens:
        # CelebA records gender as Male=1/0; translate the common Female name.
        if name.lower() == "female":
            name, value = "Male", 0.0
        elif name in selected_attrs:
            value = 1.0
        else:
            unknown.append(name)
            continue
        if name in requested and requested[name] != value:
            raise ValueError(f"Conflicting values requested for {name}")
        requested[name] = value

    unknown = set(unknown) - set(selected_attrs)
    if unknown:
        raise ValueError(f"Unknown attribute name(s): {sorted(unknown)}. "
                         f"Choose from {selected_attrs} or use Female.")

    if strict:
        attrs = np.zeros((num_samples, len(selected_attrs)), dtype=np.float32)
        for index, attr in enumerate(selected_attrs):
            if attr in requested:
                attrs[:, index] = requested[attr]
    else:
        all_attrs, _, values = _read_attr_file(attr_path)
        values = (values + 1) / 2
        indices = [all_attrs.index(attr) for attr in selected_attrs]
        candidates = values[:, indices]
        matches = np.ones(len(candidates), dtype=bool)
        for index, attr in enumerate(selected_attrs):
            if attr in requested:
                matches &= candidates[:, index] == requested[attr]
        candidates = candidates[matches]
        if len(candidates) == 0:
            raise ValueError("No CelebA label combinations match the requested attributes")
        chosen = np.random.randint(0, len(candidates), size=num_samples)
        attrs = candidates[chosen].astype(np.float32, copy=True)
    return torch.from_numpy(attrs), requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--attrs", type=str, default="",
                         help="comma-separated attribute names to turn on, e.g. 'Male,Smiling'")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--out", type=str, default="generated.png")
    parser.add_argument("--strict_attrs", action="store_true",
                        help="set every unspecified attribute to off (the old behavior)")
    parser.add_argument("--candidates_per_sample", type=int, default=64,
                        help="try this many latent samples and keep the strongest requested-attribute match")
    parser.add_argument("--top_candidates", type=int, default=1,
                        help="randomly choose among this many top matches (1 maximizes adherence)")
    parser.add_argument("--contrast", type=float, default=1.8,
                        help="contrast enhancement per face (1 disables; default 1.8 helps muted checkpoints)")
    parser.add_argument("--scale", type=int, default=4,
                        help="integer enlargement factor for the 64x64 generated faces")
    args = parser.parse_args()
    if args.num_samples < 1 or args.candidates_per_sample < 1 or args.top_candidates < 1:
        raise ValueError("sample and candidate counts must be at least 1")

    device = config.device
    G = Generator(config.latent_dim, config.num_attrs, config.g_conv_dim, config.channels).to(device)
    D = Discriminator(config.num_attrs, config.d_conv_dim, config.channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    G.eval()
    D.eval()

    candidates_per_sample = args.candidates_per_sample
    attr_vector = parse_attr_string(
        args.attrs, config.selected_attrs,
        args.num_samples * candidates_per_sample,
        config.attr_path, strict=args.strict_attrs,
    )
    attr_vector, requested = attr_vector
    attr_vector = attr_vector.to(device)
    z = torch.randn(args.num_samples * candidates_per_sample,
                    config.latent_dim, device=device)

    with torch.no_grad():
        fakes = G(z, attr_vector)
        _, attr_logits = D(fakes)
        if requested:
            requested_indices = [config.selected_attrs.index(attr) for attr in requested]
            target_values = torch.tensor(
                [requested[attr] for attr in requested], device=device,
                dtype=attr_logits.dtype,
            ).unsqueeze(0).expand(-1, -1)
            target_values = target_values.expand(len(attr_logits), -1)
            scores = -F.binary_cross_entropy_with_logits(
                attr_logits[:, requested_indices], target_values, reduction="none"
            ).sum(dim=1)
            scores = scores.view(args.num_samples, candidates_per_sample)
            top_count = min(args.top_candidates, candidates_per_sample)
            top_indices = scores.topk(top_count, dim=1).indices
            random_rank = torch.randint(top_count, (args.num_samples,), device=device)
            chosen = top_indices[torch.arange(args.num_samples, device=device), random_rank]
            best = chosen + torch.arange(args.num_samples, device=device) * candidates_per_sample
            fakes = fakes[best]
        else:
            fakes = fakes[::candidates_per_sample]
    # The trained checkpoint emits a compressed tonal range. Enhance each face
    # independently so the grid background does not affect its contrast, then
    # enlarge with Lanczos for a more useful preview. This is display
    # post-processing; it cannot recover detail absent from the checkpoint.
    if args.contrast <= 0 or args.scale < 1:
        raise ValueError("--contrast must be positive and --scale must be at least 1")
    fakes = ((fakes.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    faces = []
    for face in fakes:
        image = Image.fromarray(np.transpose(face, (1, 2, 0)), mode="RGB")
        if args.contrast != 1:
            image = ImageEnhance.Contrast(image).enhance(args.contrast)
        if args.scale != 1:
            image = image.resize((image.width * args.scale, image.height * args.scale),
                                 Image.Resampling.LANCZOS)
        faces.append(image)
    columns = min(4, len(faces))
    padding = 8
    grid = Image.new("RGB", (columns * faces[0].width + (columns + 1) * padding,
                              ((len(faces) + columns - 1) // columns) * faces[0].height +
                              ((len(faces) + columns - 1) // columns + 1) * padding),
                     (18, 18, 18))
    for index, face in enumerate(faces):
        x = padding + (index % columns) * (face.width + padding)
        y = padding + (index // columns) * (face.height + padding)
        grid.paste(face, (x, y))
    grid.save(args.out)
    if not args.strict_attrs and len(requested) < config.num_attrs:
        print("Unspecified attributes were sampled from CelebA label combinations.")
    if requested:
        print(f"Selected from the top {min(args.top_candidates, candidates_per_sample)} of "
              f"{candidates_per_sample} candidates per sample.")
    print(f"Saved {args.num_samples} samples to {args.out}")


if __name__ == "__main__":
    main()
