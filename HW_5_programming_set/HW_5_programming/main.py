import argparse
import random, math, os, numpy as np
from PIL import Image, ImageDraw
from tqdm import trange
import glob
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import argparse
import glob
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch.nn.functional as F
import numpy as np
from PIL import Image

import random
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

SHAPES  = ['circle', 'triangle', 'square']
COLORS = {
    'red'   : (230,  25,  25),
    'orange': (255, 165,   0),
    'yellow': (255, 221,   0),
    'green' : (  60, 179, 113),
    'teal'  : (  15, 200, 200),   
    'blue'  : (  15,  90, 255),  
    'purple': (145,  75, 245),
    'pink'  : (245, 105, 180),
}
COLORS_ORDER = ['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink']
COLOR_TO_IDX = {name: idx for idx, name in enumerate(COLORS_ORDER)}
SHAPE_TO_IDX = {name: idx for idx, name in enumerate(SHAPES)}
IMG_SIZE  = 64
N_SAMPLES = 32000
OUT_DIR   = 'data_folder'


def plot_samples_grid(out_dir: str = OUT_DIR,
                      display_inline: bool = True) -> None:
    """Generate an 8x3 (colors x shapes) sample grid image from the dataset."""
    output_path = 'samples_q1.png'
    BORDER_COLOR = 'white'
    BORDER_WIDTH = 1.5
    color_order = sorted(COLORS.keys())
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Directory '{out_dir}' not found. Run dataset generation first.")

    samples: Dict[str, Dict[str, Optional[str]]] = {c: {s: None for s in SHAPES} for c in color_order}
    for color in color_order:
        for shape in SHAPES:
            pattern = os.path.join(out_dir, f"*_{shape}_{color}.png")
            matches: List[str] = sorted(glob.glob(pattern))
            samples[color][shape] = matches[0] if matches else None

    rows = len(color_order)
    cols = len(SHAPES)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), facecolor='black')
    fig.patch.set_facecolor('black')
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    for i, color in enumerate(color_order):
        for j, shape in enumerate(SHAPES):
            ax = axes[i, j]
            ax.set_facecolor('black')
            path: Optional[str] = samples[color][shape]
            if path is not None and os.path.isfile(path):
                img = Image.open(path).convert('RGB')
                ax.imshow(np.asarray(img))
            else:
                ax.text(0.5, 0.5, 'N/A', color='white', fontsize=14, ha='center', va='center')
                ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER_COLOR)
                spine.set_linewidth(BORDER_WIDTH)
            if i == 0:
                ax.set_title(shape, fontsize=12)
            if j == 0:
                rgb = COLORS.get(color, (200,200,200))
                mcolor = tuple([v/255.0 for v in rgb])
                ax.set_ylabel(color, rotation=0, labelpad=40, fontsize=12, color=mcolor, va='center')

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.18, top=0.95, wspace=0.08, hspace=0.08)
    plt.savefig(output_path, dpi=150, facecolor='black')
    print(f"Saved samples grid to {output_path}")

    if display_inline:
        try:
            from IPython.display import display
            display(Image.open(output_path))
        except Exception:
            pass
    plt.close(fig)
    
def draw_shape(shape, color):
    """
        Draw a single colored shape on a black background.

        Args:
            shape: One of 'circle', 'square', or 'triangle'.
            color: RGB tuple (R, G, B) with values in [0, 255].

        Returns:
            A PIL.Image object of size (IMG_SIZE, IMG_SIZE) with the specified shape.
    """
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0,0,0))
    d   = ImageDraw.Draw(img)
    # random scale & translation
    s = random.uniform(0.4,0.8)*IMG_SIZE
    cx = random.uniform(0.3,0.7)*IMG_SIZE
    cy = random.uniform(0.3,0.7)*IMG_SIZE
    if shape=='circle':
        #### Your job 1 starts here ####
        pass
        #### Your job 1 ends here ####
    elif shape=='square':
        #### Your job 1 starts here ####
        pass
        #### Your job 1 ends here ####
    elif shape=='triangle':
        #### Your job 1 starts here ####
        pass
        #### Your job 1 ends here ####
    return img


def create_dataset():
    """
       Generate N_SAMPLES random shape images and save them to OUT_DIR.

       Each filename encodes the shape and color, e.g. '00001_circle_red.png'.
       Also tries to create a small visualization grid after generation.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    iterator = trange(N_SAMPLES, desc='Generating', unit='img')
    for i in iterator:
        shape = random.choice(SHAPES)
        color_name, base_color = random.choice(list(COLORS.items()))
        img = draw_shape(shape, base_color)
        fname = f'{OUT_DIR}/{i:05}_{shape}_{color_name}.png'
        img.save(fname)
    try:
        plot_samples_grid(out_dir=OUT_DIR)
    except Exception as exc:
        print(f"Warning: could not plot samples grid ({exc})")


def parse_labels(path: str) -> Tuple[int, int]:
    """
       Parse shape and color labels from a filename.

       Expected filename format:
           XXXXX_shape_color.png
       e.g.,
           '00001_circle_red.png'

       Args:
           path: Full path to the image file.

       Returns:
           (shape_idx, color_idx): integer indices using SHAPE_TO_IDX and COLOR_TO_IDX.
    """
    base = os.path.basename(path)
    try:
        _, shape_name, color_ext = base.split('_')
    except ValueError as exc:
        raise ValueError(f"Unexpected filename format: {base}") from exc
    color_name = color_ext.split('.')[0]
    return SHAPE_TO_IDX[shape_name], COLOR_TO_IDX[color_name]


class ColorShapeDataset(Dataset):
    """
        PyTorch Dataset for the colored-shape images.

        Each item is:
            img:   FloatTensor of shape (3, IMG_SIZE, IMG_SIZE)
            shape: int index for the shape class
            color: int index for the color class
    """
    def __init__(self, all_paths, transform: Optional[transforms.Compose] = None):
        """
                Args:
                    all_paths: List of image file paths.
                    transform: Optional torchvision transform to apply to each PIL image.
        """
        #### Your job 2 starts here ####
        pass
        #### Your job 2 ends here ####

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        """
                Retrieve a single (image, shape_label, color_label) tuple.

                Args:
                    idx: Integer index into the dataset.

                Returns:
                    img: Tensor of shape (3, IMG_SIZE, IMG_SIZE), in [0, 1].
                    shape_idx: Shape label as an int.
                    color_idx: Color label as an int.
        """
        #### Your job 2 starts here ####
        # The line below is just a placeholder. Remove it and implement the method.
        img, shape_idx, color_idx = None, None, None
        #### Your job 2 ends here ####
        return img, shape_idx, color_idx


def build_transform() -> transforms.Compose:
    ops = [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ]
    return transforms.Compose(ops)


class ConvAutoencoder(nn.Module):
    """
        Convolutional autoencoder for 64x64 RGB images.

        The encoder downsamples the input into a compact latent tensor.
        The decoder upsamples back to the original resolution.
    """
    def __init__(self):
        super().__init__()
        enc_channels = [3, 16, 32, 64, 64, 128, 128]
        encoder_layers = []
        #### Your job 2 starts here ####
        pass
        #### Your job 2 ends here ####
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.enc_out_channels = enc_channels[-1]  # 128

        dec_channels = [128, 128, 64, 64, 32, 16, 3]
        decoder_layers = []
        #### Your job 2 starts here ####
        pass
        #### Your job 2 ends here ####
        self.decoder_layers = nn.ModuleList(decoder_layers)

    def encode(self, x: torch.Tensor, return_intermediate: bool = False):
        """
               Run the encoder on input x.

               Args:
                   x: Input tensor of shape (B, 3, H, W).
                   return_intermediate: If True, also return a list of intermediate
                       encoder outputs (one per encoder layer).

               Returns:
                   latent: Final encoder output tensor (B, C_latent, H_latent, W_latent).
                   intermediates (optional): List of tensors (one per encoder layer).
        """
        #### Your job 2 starts here ####
        # The line below is just a placeholder. Remove it and implement the method.
        latent, intermediates = None, None
        #### Your job 2 ends here ####
        if return_intermediate:
            return latent, intermediates
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
                Run the decoder starting from a latent tensor.

                Args:
                    latent: Encoder output tensor (B, C_latent, H_latent, W_latent).

                Returns:
                    out: Reconstructed image tensor of shape (B, 3, IMG_SIZE, IMG_SIZE).
        """
        #### Your job 2 starts here ####
        # The line below is just a placeholder. Remove it and implement the method.
        out = None
        #### Your job 2 ends here ####
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x, return_intermediate=False)
        return self.decode(latent)


def build_dataloader(data_dir: str,
                     batch_size: int,
                     num_workers: int,
                     shuffle: bool) -> DataLoader:
    """
        Build a DataLoader over ColorShapeDataset for a given directory.

        Args:
            data_dir: Directory containing '*.png' images.
            batch_size: Batch size for the DataLoader.
            num_workers: Number of worker processes for data loading.
            shuffle: Whether to shuffle the dataset each epoch.

        Returns:
            A PyTorch DataLoader yielding (img, shape_label, color_label).
    """
    all_paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))
    #### Your job 2 starts here ####
    # The line below is just a placeholder. Remove it and implement the method.
    loader = None
    #### Your job 2 ends here ####
    return loader


def train_autoencoder(model: ConvAutoencoder, train_loader: DataLoader,
                       device: torch.device,
                      epochs: int, lr: float, weight_decay: float,
                      checkpoint_path: str) -> None:
    """
        Train the ConvAutoencoder on the provided dataset.

        Args:
            model: ConvAutoencoder instance.
            train_loader: DataLoader over the training set.
            device: torch.device ('cpu' or 'cuda').
            epochs: Number of training epochs.
            lr: Learning rate.
            weight_decay: Weight decay (L2 regularization).
            checkpoint_path: Where to save the final model checkpoint.

        Side-effect:
            Saves {'model_state': model.state_dict()} to checkpoint_path.
    """
    #### Your job 2 starts here ####
    pass
    #### Your job 2 ends here ####
    torch.save({'model_state': model.state_dict()}, checkpoint_path)
    # ------------------------------------------------------------------


def load_raw_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    return transforms.ToTensor()(img)


def flatten_feature(t: torch.Tensor) -> torch.Tensor:
    return torch.flatten(t, start_dim=1)


def collect_embeddings(model: ConvAutoencoder, loader: DataLoader,
                       device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
        Run the encoder over a dataset to collect features for each layer.

        Args:
            model: Trained ConvAutoencoder.
            loader: DataLoader yielding (img, shape_label, color_label).
            device: torch.device.

        Returns:
            features: Dict mapping
                - 'pixels'     -> raw pixel tensors (flattened or feature-like)
                - 'layer_k'    -> encoder outputs for each layer k
            all_shapes: LongTensor of shape (N,) containing shape labels.
            all_colors: LongTensor of shape (N,) containing color labels.
    """
    model.eval()
    # initialize containers for raw pixels and each encoder layer output
    features: Dict[str, List[torch.Tensor]] = {'pixels': []}
    num_layers = len(model.encoder_layers)
    for layer_idx in range(1, num_layers + 1):
        features[f'layer_{layer_idx}'] = []
    shapes, colors = [], []

    with torch.no_grad():
        #### Your job 3 starts here ####
        pass
        #### Your job 3 ends here ####

    out_features = {name: torch.cat(tensors, dim=0) for name, tensors in features.items()}

    all_shapes = torch.cat(shapes)
    all_colors = torch.cat(colors)
    return out_features, all_shapes, all_colors


def cross_nearest_neighbor_accuracy(
    emb_query: torch.Tensor,
    labels_query: torch.Tensor,
    emb_ref: torch.Tensor,
    labels_ref: torch.Tensor,
) -> float:
    """
    For each query embedding (test), find its nearest neighbor in the ref set (train)
    and compute label accuracy.

    Args:
        emb_query: Query embeddings of shape (N_query, D).
        labels_query: Labels for query embeddings of shape (N_query,).
        emb_ref: Reference embeddings of shape (N_ref, D).
        labels_ref: Labels for reference embeddings of shape (N_ref,).

    Returns:
        Accuracy in percentage as a float.
    """
    # L2-normalize both sets
    q = F.normalize(emb_query, dim=1)
    r = F.normalize(emb_ref,   dim=1)

    # dist: (N_query, N_ref)
    dist = torch.cdist(q, r)
    nn_idx = torch.argmin(dist, dim=1)          # index into ref set

    pred_labels = labels_ref[nn_idx]
    correct = (labels_query == pred_labels).float().mean().item()
    return correct * 100.0


def save_nearest_neighbor_grid_cross(
    query_dataset,
    query_embeddings: torch.Tensor,
    ref_dataset,
    ref_embeddings: torch.Tensor,
    out_path: str,
    num_queries: int = 5,
    neighbors: int = 5,
    seed: Optional[int] = None,
) -> None:
    """
        Build and save a grid of nearest neighbors across two datasets.

        Each row in the grid:
            [ query | neighbor_1 | neighbor_2 | ... neighbor_K ]

        Args:
            query_dataset: Dataset providing __getitem__(idx) -> (img, shape, color) for queries.
            query_embeddings: Embeddings for queries, shape (N_query, D).
            ref_dataset: Dataset for reference images.
            ref_embeddings: Embeddings for references, shape (N_ref, D).
            out_path: Path to save the resulting PNG image.
            num_queries: Number of query images to randomly sample.
            neighbors: Number of nearest neighbors per query.
            seed: Optional random seed for reproducibility.
    """

    #### Your job 3 starts here ####
    dist = None
    #### Your job 3 ends here ####

    Nq = len(query_dataset)
    neighbors   = min(neighbors, max(1, len(ref_dataset) - 1))

    rng = random.Random(seed)
    query_indices = rng.sample(range(Nq), k=num_queries)

    # ---- 2. Build rows: [query (test) | white separator | neighbors (train)...] ----
    rows = []
    for q_idx in query_indices:
        #### Your job 3 starts here ####
        nn_indices = None
        #### Your job 3 ends here ####
        # query from test set
        query_img = load_raw_tensor(query_dataset.paths[q_idx])  # (3, H, W)

        # neighbors from train set
        nn_imgs = [load_raw_tensor(ref_dataset.paths[i]) for i in nn_indices]

        sep = torch.ones_like(query_img)  # full-size white tile

        row_imgs = [query_img, sep] + nn_imgs
        rows.append(torch.stack(row_imgs))  # (neighbors+2, 3, H, W)

    all_rows = torch.cat(rows, dim=0)

    # ---- 3. Make grid with white padding ----
    ncols = neighbors + 2  # query + separator + neighbors
    grid = make_grid(all_rows, nrow=ncols, padding=2, pad_value=1.0)

    # ---- 4. Plot on white background and add titles ----
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_np)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    query_x = (0.5 / ncols)
    neigh_x = (2 + neighbors / 2.0) / ncols

    ax.text(query_x, 1.02, "query (test)",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=14)
    ax.text(neigh_x, 1.02, "nearest neighbors (train)",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=14)

    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Saved cross-dataset nearest-neighbor grid to {out_path}")
    return


def plot_layerwise_nn(color_accs, shape_accs, out_path: str):
    """
    Plot layer-wise 1-NN color and shape accuracy.

    Args:
        color_accs: List of color accuracies (pixels + layers).
        shape_accs: List of shape accuracies (pixels + layers).
        out_path: Path to save the PNG plot.

    Produces:
        A line plot similar to the reference figure, with x-axis being layer index
        (including 'pixels' as layer 0).
    """
    #### Your job 3 starts here ####
    pass
    #### Your job 3 ends here ####


def evaluate(model: ConvAutoencoder,
             train_dir: str,
             test_dir: str,
             device: torch.device,
             batch_size: int,
             num_workers: int,
             output_dir: str) -> None:
    """
        Evaluate learned representations using cross-dataset nearest neighbors.

        Args:
            model: Trained ConvAutoencoder.
            train_dir: Directory with training images (for reference set).
            test_dir: Directory with test images (for query set).
            device: torch.device.
            batch_size: Batch size for feature extraction.
            num_workers: Number of workers for data loading.
            output_dir: Directory to save figures.

        Side-effects:
            - Prints layer-wise color and shape accuracy.
            - Saves 'layerwise_nn_acc.png' and 'nearest_neighbors.png' in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Train set: used as the nearest-neighbor database ----
    all_train_paths = sorted(glob.glob(os.path.join(train_dir, '*.png')))
    rng = random.Random(0)
    subset_indices = rng.sample(range(len(all_train_paths)), k=3000)
    subset_indices.sort()
    train_paths = [all_train_paths[i] for i in subset_indices]
    print(f"Using a subset of {len(train_paths)} / {len(all_train_paths)} train images for kNN.")

    train_dataset = ColorShapeDataset(train_paths, transform=build_transform())
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)
    # ---- Test set: used as queries ----
    test_paths = sorted(glob.glob(os.path.join(test_dir, '*.png')))
    test_dataset = ColorShapeDataset(test_paths, transform=build_transform())
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    # ---- Collect embeddings on train and test ----
    features_train, shapes_train, colors_train = collect_embeddings(model, train_loader, device)
    features_test,  shapes_test,  colors_test  = collect_embeddings(model, test_loader,  device)

    # ---- Cross 1-NN accuracy: test → train ----
    layer_order = ['pixels'] + [f'layer_{i}' for i in range(1, len(model.encoder_layers)+1)]

    color_accs = []
    shape_accs = []

    for layer_name in layer_order:
        #### Your job 3 starts here ####
        pass
        #### Your job 3 ends here ####

    # ---- plot figure like the reference ----
    plot_layerwise_nn(color_accs, shape_accs,
                      out_path=os.path.join(output_dir, "layerwise_nn_acc.png"))

    # ---- Use last encoder layer (bottleneck) for the NN grid ----
    last_layer_name = f'layer_{len(model.encoder_layers)}'
    save_nearest_neighbor_grid_cross(
        query_dataset=test_dataset,
        query_embeddings=features_test[last_layer_name],
        ref_dataset=train_dataset,
        ref_embeddings=features_train[last_layer_name],
        out_path=os.path.join(output_dir, 'nearest_neighbors.png'),
        num_queries=5,
        neighbors=5,
        seed=0,
    )


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.job_number == 1:
        create_dataset()

    if args.job_number == 2:
        train_loader = build_dataloader(args.data_dir, args.batch_size,
                                        args.num_workers, shuffle=True)
        model = ConvAutoencoder().to(device)
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(f"{name:40s} {p.numel():10d}")
        print("Total trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        checkpoint_path = os.path.join(args.output_dir, 'autoencoder.pt')
        train_autoencoder(model, train_loader, device,
                          args.epochs, args.lr, args.wd, checkpoint_path)
    if args.job_number == 3:
        model = ConvAutoencoder().to(device)
        checkpoint_path = os.path.join(args.output_dir, 'autoencoder.pt')
        model.load_state_dict(torch.load(checkpoint_path)['model_state'])
        evaluate(model, args.data_dir, args.test_data_dir, device, args.batch_size, args.num_workers, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HW5")
    parser.add_argument('--job_number', default=3, type=int)
    parser.add_argument('--data-dir', type=str, default='data_folder', help='Path to toy shapes images.')
    parser.add_argument('--test-data-dir', type=str, default='data_folder_test', help='Path to test shapes images.')
    parser.add_argument('--output-dir', type=str, default='ae_runs', help='Directory to store checkpoints/plots.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    main(args)