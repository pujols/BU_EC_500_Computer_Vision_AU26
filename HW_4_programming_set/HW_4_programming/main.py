# Homework 3 - Questions 1, 2, 3, 4 & 5
# Region Proposal Network (RPN) - GT Tensor Generation, Decoding, Anchor-based Encoding/Decoding, and Non-Maximum Suppression

import argparse
import numpy as np
import os
import os.path as osp
import json
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

PATCH_SIZE = 20
TARGET_SIZE = 200
GRID_NUMBER = int(TARGET_SIZE / PATCH_SIZE)
THRESHOLD = 0.65  # Confidence threshold for 'Existence: Yes'


def resize_image_and_boxes(image_path, boxes):
    """
    Resizes the input image to TARGET_SIZE x TARGET_SIZE and scales bounding boxes
    already defined in the bottom-left origin (x=col, y=row from bottom).

    Args:
        image_path (str): Path to the image file.
        boxes (list): List of bounding boxes [x, y, w, h] in bottom-left coordinate system.

    Returns:
        image (PIL.Image): Resized image.
        resized_boxes (list): List of resized boxes [x, y, w, h].
    """
    image = Image.open(image_path).convert("RGB")
    resized_boxes = []

    #### Your job 1 starts here ####

    #### Your job 1 ends here ####

    return image, resized_boxes


def compute_patch_grid():
    """
    Creates a grid of patches over a TARGET_SIZE x TARGET_SIZE image using GRID_NUMBER and PATCH_SIZE,
    assuming a coordinate system where (0, 0) is at the bottom-left.

    Returns:
        grid (list): List of tuples (i, j, x_center, y_center, w, h) for each patch, where:
            - (i, j): grid column and row index (with (0,0) at bottom-left)
            - (x_center, y_center): center coordinates of the patch
            - (w, h): width and height of the patch
    """
    grid = []

    for i in range(GRID_NUMBER):        # i: column index (x-direction)
        for j in range(GRID_NUMBER):    # j: row index (y-direction, bottom-up)

            #### Your job 1 starts here
            pass
            #### Your job 1 ends here ####

    return grid


def boxes_overlap(box1, box2):
    """
    Checks whether two boxes overlap (non-zero intersection area),
    assuming (0, 0) is at the bottom-left and boxes are in center format.

    Args:
        box1, box2 (list or tuple): Bounding boxes in 
            [x_center, y_center, width, height] format, using bottom-left origin.

    Returns:
        bool: True if boxes overlap, False otherwise.
    """
    result = True

    #### Your job 1 starts here ####

    #### Your job 1 ends here ####

    return result


def generate_gt_tensors(boxes):
    """
    Generates ground-truth existence and location tensors for a resized image.

    Args:
        boxes (list): List of ground-truth bounding boxes in the format [x_center, y_center, width, height],
                      assuming a coordinate system with (0,0) at the bottom-left.

    Returns:
        existence (ndarray): Tensor of shape (GRID_NUMBER, GRID_NUMBER, 2), where
                             existence[j, i, 0] = 1 if an object exists in patch (i, j),
                             existence[j, i, 1] = 1 otherwise (i.e., background patch).
        location (ndarray): Tensor of shape (GRID_NUMBER, GRID_NUMBER, 4), where
                            location[j, i] = [x_center, y_center, width, height] of a GT box
                            assigned to patch (i, j), if any.
    """
    # Initialize the existence tensor with shape (grid_height, grid_width, 2)
    # It will hold one-hot encoded values: [1, 0] for object, [0, 1] for background
    existence = np.zeros((GRID_NUMBER, GRID_NUMBER, 2))

    # Initialize the location tensor to store box details for positive patches
    location = np.zeros((GRID_NUMBER, GRID_NUMBER, 4))

    # Generate the patch grid using bottom-left (0, 0) origin, returning:
    # (i, j, x_center, y_center, w, h) for each patch
    grid = compute_patch_grid()

    #### Your job 1 starts here ####

    #### Your job 1 ends here ####

    return existence, location


def decode_tensors(existence, location, original_w, original_h, threshold=THRESHOLD):
    """
    Converts tensors back into bounding boxes by thresholding on the confidence map.

    Args:
        existence (ndarray): Existence tensor from model output.
        location (ndarray): Location tensor from model output.
        original_w (int): Width of the original image.
        original_h (int): Height of the original image.
        threshold (float): Confidence threshold to keep predicted boxes.

    Returns:
        boxes (list): List of decoded boxes in [cx, cy, w, h, confidence].
    """
    boxes = set([])
    #### Your job 2 starts here ####

    #### Your job 2 ends here ####
    return list(boxes)


def tensor_to_image_name(tensor_file):
    """
    Extracts base image name from a tensor file name.

    Args:
        tensor_file (str): Filename of a tensor.

    Returns:
        str: Base name of the original image.
    """
    base = osp.basename(tensor_file)
    return base.replace("1_Existence_tensor_", "").replace(".npz", "")


def compute_iou(box1, box2):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes.

    Args:
        box1 (list or tuple): The first box in [cx, cy, w, h] format,
            where (cx, cy) is the center of the box, and (w, h) are width and height.
        box2 (list or tuple): The second box in [cx, cy, w, h] format.

    Returns:
        float: IoU score between box1 and box2, ranging from 0 (no overlap)
               to 1 (perfect overlap).
    """
    result = True

    #### Your job 3 starts here ####

    #### Your job 3 ends here ####

    return result

def encode_anchors_multi_gt(resized_boxes, grid, anchor_shapes):
    """
    Encode supervision for all anchors when there may be multiple GT boxes
    per image and multiple GT boxes overlapping the same patch. This routine follows
    the policy:

        For each GT:
            For each patch that overlaps this GT:
                choose the anchor (among all anchor_shapes) that yields the **largest IoU**
                with this GT at this patch;
                claim that (patch, anchor) if it is unused OR if this GT's IoU is larger
                than the IoU of the GT that previously claimed it.

    Parameters
    ----------
        resized_boxes (List[List[float]]): A list of ground truth boxes, where each box is in
            [cx, cy, w, h] format and has been resized to match the current image dimensions.

        grid (List[Tuple[int, int, float, float, float, float]]): A list representing the patch grid.
            Each entry is a tuple (i, j, x_center, y_center, w, h), where (i, j) are grid indices and
            (x_center, y_center, w, h) define the center coordinates and size of each patch.

        anchor_shapes : List[Tuple[float, float]]  List of tuples (ws, hs) representing the width and height scaling
            factors (relative to the patch size) for the current anchor.

    Returns
    -------
    existence_list : List[np.ndarray]
        A list of length K, where each tensor of shape (GRID_NUMBER, GRID_NUMBER, 2), where each entry is [1, 0]
            if the best-matching anchor overlaps with a ground truth box, otherwise remains [0, 0]).

    location_list : List[np.ndarray]
        A list of length K, where tensor of shape (GRID_NUMBER, GRID_NUMBER, 4), containing the offsets
            [dx, dy, dw, dh] between the matched anchor and the ground truth box for each patc.

    """

    K = len(anchor_shapes)
    existence_list = [np.zeros((GRID_NUMBER, GRID_NUMBER, 2), dtype=np.float32) for _ in range(K)]
    location_list  = [np.zeros((GRID_NUMBER, GRID_NUMBER, 4), dtype=np.float32) for _ in range(K)]

    # Track current best score per (anchor k, patch j,i).
    best_iou = np.full((K, GRID_NUMBER, GRID_NUMBER), -1.0, dtype=np.float32)

    # Loop GT -> patch (the important ordering)
    for gt_idx, gt in enumerate(resized_boxes):
        for i, j, x_center, y_center, w, h in grid:
            # Skip if the patch doesn't overlap with this GT
            if not boxes_overlap(gt, [x_center, y_center, w, h]):
                continue

            #### Your job 3 starts here ####
            # Among all anchors, find the ONE with the largest IoU for this (GT, patch)


            # once you find the best anchor
            #    - If unused (best_iou = -1), this GT claims it.
            #    - If already claimed by another GT, replace only if this IoU is larger.
            #    - Update existence_list and location_list

            #### Your job 3 ends here ####
    return existence_list, location_list




def process_anchor_encoding(args, anchor_shapes, image_dict, ann_dict):
    """
    Performs anchor-based encoding of ground-truth boxes into tensors.

    Args:
        args (argparse.Namespace): Command-line arguments containing options like:
            - args.image_folder: path to the folder containing images.
            - args.save (bool): whether to save the output tensors to disk.
            - args.display (bool): whether to visualize the encoded results.
        anchor_shapes (list of tuples): List of (width_scale, height_scale) pairs
            representing anchor box shapes relative to the patch size.
        image_dict (dict): Dictionary mapping image IDs to metadata (e.g., file_name).
        ann_dict (dict): Dictionary mapping image IDs to lists of ground truth boxes
            in [cx, cy, w, h] format.

    Outputs:
        - If args.save is True, saves:
            - Existence tensors: (GRID_NUMBER, GRID_NUMBER, 2)
            - Location tensors: (GRID_NUMBER, GRID_NUMBER, 4)
          to the "result" folder, one file per anchor per image.
        - If args.display is True, visualizes the predicted boxes against ground truth.
    """

    # Generate the patch grid once for all images
    grid = compute_patch_grid()

    # Iterate over all images in the dataset
    for img_id, image_info in image_dict.items():
        if img_id not in ann_dict:
            continue # Skip images without ground-truth annotations

        # Get the list of ground truth boxes for this image
        gt_boxes = ann_dict[img_id]

        # Construct full image path and extract file name
        file_name = image_info['file_name']
        image_path = osp.join(args.image_folder, file_name)

        # Resize the image and scale boxes accordingly
        image, resized_boxes = resize_image_and_boxes(image_path, gt_boxes)
        image_name = osp.splitext(file_name)[0] # Filename without extension


        existence_list, location_list = encode_anchors_multi_gt(resized_boxes, grid, anchor_shapes)

        # Loop over each anchor shape (width_scale, height_scale)
        for k, anchor_shape in enumerate(anchor_shapes):
            ws, hs = anchor_shape

            existence = existence_list[k]
            location = location_list[k]

            # Save encoded tensors to disk if required
            if args.save:
                np.savez(osp.join("result", f"3_Anchor_{k}_Existence_tensor_{image_name}.npz"), existence=existence)
                np.savez(osp.join("result", f"3_Anchor_{k}_Location_tensor_{image_name}.npz"), location=location)

            # Optional visualization of matched positive boxes
            if args.display:
                bboxes = set([])
                for i in range(GRID_NUMBER):
                    for j in range(GRID_NUMBER):
                        if existence[j, i, 0] > 0:
                            # Compute patch center coordinates
                            patch_cx = i * PATCH_SIZE + PATCH_SIZE / 2
                            patch_cy = j * PATCH_SIZE + PATCH_SIZE / 2

                            # Get the predicted offset from location tensor
                            dx, dy, dw, dh = location[j, i]

                            # Reconstruct anchor box dimensions
                            anchor_w = PATCH_SIZE * ws
                            anchor_h = PATCH_SIZE * hs
                            
                            # Decode final predicted box (in [cx, cy, w, h])
                            cx = patch_cx + dx
                            cy = patch_cy + dy
                            w = anchor_w + dw
                            h = anchor_h + dh
                            bboxes.add((round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)))

                # Display boxes for predicted positive anchors
                bboxes = list(bboxes)
                print( f"\nAnchor {k} Positive Boxes for {image_name}:")
                for bbox in bboxes:
                    print(f"[x_center={bbox[0]:.10f}, y_center={bbox[1]:.10f}, width={bbox[2]:.10f}, height={bbox[3]:.10f}]")
                display(image, bboxes, title=f"Anchor {k} Positive Boxes: {image_name}")
                

def decode_anchor_grid_predictions(existence, location, anchor_shape, original_size, threshold):
    """
    Decodes predicted anchor-based bounding box outputs from a grid-based format.

    This function takes predicted existence/confidence scores and location offsets 
    from a fixed grid of anchor boxes and converts them into bounding box coordinates 
    in the original image space. Only boxes with confidence scores above the specified 
    threshold are returned.

    Args:
        existence (np.ndarray): A (GRID_NUMBER, GRID_NUMBER, 2) array containing confidence 
                                scores for each grid cell.
        location (np.ndarray): A (GRID_NUMBER, GRID_NUMBER, 4) array containing predicted 
                               offsets (dx, dy, dw, dh) for each grid cell.
        anchor_shape (Tuple[float, float]): The normalized width and height (ws, hs) of the 
                                            anchor box for this prediction level.
        original_size (Tuple[int, int]): The original width and height of the input image.
        threshold (float): Confidence threshold for including a bounding box.

    Returns:
        Set[Tuple[float, float, float, float, float]]: A set of predicted bounding boxes 
        in the format (cx, cy, w, h, confidence), all scaled to the original image dimensions.
    """
    boxes = set()
    original_w, original_h = original_size
    ws, hs = anchor_shape

    #### Your job 4 starts here ####

    #### Your job 4 ends here ####
    return boxes


def decode_anchor_outputs(args):
    """
    Decodes anchor-based existence and location tensors to produce bounding box proposals.

    Args:
        args (argparse.Namespace): Command-line arguments with the following expected attributes:
            - args.anchor_file (str): Path to JSON file defining anchor shapes per index.
            - args.tensor_folder (str): Folder containing existence and location tensors.
            - args.image_folder (str): Folder containing input images.
            - args.threshold (float): Confidence threshold for including a box.
            - args.display (bool): Whether to visualize boxes on images.
            - args.job_number (int): Used to control whether NMS is applied and title of display.

    Outputs:
        For each image base name:
            - Prints decoded bounding box proposals with their confidence scores.
            - Optionally displays the boxes drawn on the original image.
    """
    # Load anchor shapes from JSON, each indexed by anchor level (k)
    with open(args.anchor_file, 'r') as f:
        anchor_shapes = json.load(f)

    # List all tensor files in the folder
    tensor_files = os.listdir(args.tensor_folder)

    # Extract all anchor indices (e.g., 0, 1, 2) based on filename patterns
    anchor_indices = sorted({int(f.split('_')[2]) for f in tensor_files if f.startswith("3_Anchor")})
    
    # Extract all unique base names (e.g., image filenames without extension)
    base_names = sorted(set(
        f.split("_")[-1].replace(".npz", "")
        for f in tensor_files if f.startswith("3_Anchor")
    ))

    # Loop through each image base name
    for base in base_names:
        image_path = osp.join(args.image_folder, base + ".jpg")
        image = Image.open(image_path).convert("RGB")

        boxes = set([]) # Set to hold unique decoded bounding boxes

        # Process tensors for each anchor level (k)
        for k in anchor_indices:
            exist_path = osp.join(args.tensor_folder, f"3_Anchor_{k}_Existence_tensor_{base}.npz")
            loc_path = osp.join(args.tensor_folder, f"3_Anchor_{k}_Location_tensor_{base}.npz")
            existence = np.load(exist_path)['existence']
            location = np.load(loc_path)['location']

            # Decode predictions for this anchor index and accumulate all boxes
            boxes |= decode_anchor_grid_predictions(existence, location, anchor_shapes[k], image.size, args.threshold)

        boxes = list(boxes)  # Convert to list for further processing

        # Optionally apply Non-Maximum Suppression if job_number == 5
        if args.job_number == 5:
            boxes = non_max_suppression([b[:5] for b in boxes], threshold=THRESHOLD)
        
        # Print all decoded (or post-NMS) boxes for the current image
        print(f"\nDecoded Anchor Boxes for {base}:")
        for b in boxes:
            print(f"[x_center={b[0]:.1f}, y_center={b[1]:.1f}, width={b[2]:.1f}, height={b[3]:.1f}, conf={b[4]:.2f}]")

        # Optional visualization of the boxes
        if args.display:
            title = f"NMS Final Boxes - {base}" if args.job_number == 5 else f"Anchor Decoded Boxes - {base}"
            display(image, boxes, title=title)


def non_max_suppression(boxes, threshold=THRESHOLD):
    """
    Applies Non-Maximum Suppression (NMS) to remove redundant overlapping bounding boxes.

    This function helps reduce duplicate detections by keeping only the most confident 
    bounding box among overlapping ones. It iteratively selects the box with the highest 
    confidence score and removes all other boxes that have high overlap (IoU) with it.

    Args:
        boxes (list): A list of bounding boxes, each represented as a 5-element list or tuple:
                      [cx, cy, w, h, confidence], where (cx, cy) is the center,
                      (w, h) are width and height, and `confidence` is the predicted score.
        threshold (float): The Intersection-over-Union (IoU) threshold used to suppress
                           overlapping boxes. If IoU between two boxes exceeds this value,
                           the box with lower confidence is removed.

    Returns:
        keep (list): A list of filtered bounding boxes after NMS, retaining only the
                     most confident non-overlapping predictions.
    """
    if not boxes:
        return []

    keep = []
    #### Your job 5 starts here ####

    #### Your job 5 ends here ####
    return keep


def parse_coco_annotations(annotation_file):
    """
    Reads a COCO annotation file and returns two dictionaries:
      1) image_dict: keyed by image_id, each value is a dict containing:
         - 'file_name'
         - 'width'
         - 'height'
      2) ann_dict: keyed by image_id, each value is a list of bounding boxes (bboxes).
    """
    with open(annotation_file, 'r') as f:
        coco = json.load(f)

    # Build the image dictionary with file_name, width, and height
    image_dict = {}
    for img in coco['images']:
        image_dict[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # Build the annotation dictionary with lists of bboxes
    ann_dict = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_dict:
            ann_dict[img_id] = []
        ann_dict[img_id].append(change_coordinates(ann['bbox'], image_dict[img_id]))

    return image_dict, ann_dict


def change_coordinates(bbox, image_dict):
    """
    Transform a bounding box from a top-left, (row=x, col=y) system
    to a bottom-left, (col=x, row=y) system.

    Args:
        bbox (list): [x_min, y_min, w, h] in old system
        image_dict (dict): must have 'width' (W) and 'height' (H)

    Returns:
        list: [new_x_center, new_y_center, new_w, new_h] in the new coordinate system
    """
    W = image_dict['width']
    H = image_dict['height']
    x_min, y_min, w, h = bbox

    new_x_min = x_min
    new_y_min = H - y_min - h  # "shift" so that bottom is now zero
    new_w = w
    new_h = h

    new_x_center = new_x_min + new_w/2
    new_y_center = new_y_min + new_h/2

    return [new_x_center, new_y_center, new_w, new_h]


def display(image, boxes, title=""):
    """
    Displays an image with overlaid bounding boxes and a flipped coordinate system.

    The image is shown in its normal orientation (origin at top-left), but the 
    y-axis ticks and coordinate display are adjusted to mimic a Cartesian coordinate 
    system with the origin at the bottom-left. This helps interpret coordinates in 
    a more intuitive bottom-up manner while keeping the image unchanged.

    Args:
        image (PIL.Image): The input image to display.
        boxes (List[Tuple[float]]): A list of bounding boxes in (x_center, y_center, width, height) format.
        title (str): The title of image to display

    Behavior:
        - Draws bounding boxes as red rectangles based on provided coordinates.
        - Adjusts y-axis ticks and mouse hover display to simulate (0,0) at bottom-left.
        - Keeps the image visually unflipped (standard top-left origin).
    """
    fig, ax = plt.subplots()
    ax.imshow(image)

    img_w, img_h = image.size
    
    for b in boxes:
        conf = None
        if len(b) == 4:
            x_center, y_center, w, h = b
        else:
            x_center, y_center, w, h, conf = b
        x_min = x_center - w/2
        y_min = y_center - h/2

        new_x_min = x_min
        new_y_min = img_h - y_min - h 
            
        rect = plt.Rectangle((new_x_min, new_y_min), w, h, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if conf is not None:
            ax.text(new_x_min, new_y_min, f"{conf:.2f}", color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    # Show image normally, but make axis ticks appear as if origin is bottom-left
    ax.set_xlim([0, img_w])
    ax.set_ylim([img_h, 0])  # y increases bottom-up now

    # Fix tick labels manually so 0 is at bottom
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f"{int(img_h - y)}" for y in y_ticks])  # Fake flip
    # Update coordinate formatter to flip y in the coordinate display
    def format_coord(x, y):
        flipped_y = img_h - y
        return f"x={x:.1f}, y={flipped_y:.1f}"

    ax.format_coord = format_coord

    plt.title(title)
    plt.show()


def main(args):
    """
    Main execution entry for the pipeline.

    job_number:
        1: Generate GT tensors (resized images + patch-based encoding).
        2: Decode GT tensors back to bounding boxes.
        3: Perform anchor-based encoding of ground-truth annotations.
        4: Decode anchor-based outputs (without NMS).
        5: Decode anchor-based outputs with NMS applied.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    os.makedirs('result', exist_ok=True)
    if args.job_number == 1:
        image_dict, ann_dict = parse_coco_annotations(args.annotation_file)

        for img_id, image_info in image_dict.items():
            if img_id not in ann_dict:
                continue
            file_name = image_info['file_name']
            boxes = ann_dict[img_id]
            image_path = osp.join(args.image_folder, file_name)
            image, resized_boxes = resize_image_and_boxes(image_path, boxes)
            existence, location = generate_gt_tensors(resized_boxes)
            image_name = osp.splitext(file_name)[0]

            if args.save:
                np.savez(osp.join("result", f"1_Existence_tensor_{image_name}.npz"), existence=existence)
                np.savez(osp.join("result", f"1_Location_tensor_{image_name}.npz"), location=location)

            if args.display:
                display(image, resized_boxes, title=f"Resized GT: {image_name}")
                
    elif args.job_number == 2:
        existence_files = [f for f in os.listdir(args.tensor_folder) if f.startswith("1_Existence_tensor")]

        for exist_file in existence_files:
            name = tensor_to_image_name(exist_file)
            exist_path = osp.join(args.tensor_folder, f"1_Existence_tensor_{name}.npz")
            loc_path = osp.join(args.tensor_folder, f"1_Location_tensor_{name}.npz")
            image_path = osp.join(args.image_folder, name + ".jpg")

            existence = np.load(exist_path)['existence']
            location = np.load(loc_path)['location']

            image = Image.open(image_path).convert("RGB")
            original_w, original_h = image.size
            boxes = decode_tensors(existence, location, original_w, original_h, threshold=args.threshold)

            print(f"\nDecoded boxes for {name}:")
            for b in boxes:
                print(f"[x_center={b[0]:.1f}, y_center={b[1]:.1f}, width={b[2]:.1f}, height={b[3]:.1f}, conf={b[4]:.2f}]")

            if args.display:
                display(image, boxes, title=f"Decoded Proposals - {name}")

    elif args.job_number == 3:
        print("Running advanced anchor-based encoding...")

        with open(args.anchor_file, 'r') as f:
            anchor_shapes = json.load(f)

        image_dict, ann_dict = parse_coco_annotations(args.annotation_file)

        process_anchor_encoding(args, anchor_shapes, image_dict, ann_dict)

    elif args.job_number == 4:
        print("Running advanced anchor-based decoding...")
        decode_anchor_outputs(args)

    elif args.job_number == 5:
        print("Running advanced anchor-based decoding with NMS...")
        decode_anchor_outputs(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_number', type=int, required=True, help="1: Generate GT tensors, 2: Decode proposals, 3: Anchor-based encoding")
    parser.add_argument('--image_folder', type=str, help="Folder containing input images")
    parser.add_argument('--annotation_file', type=str, help="COCO-format annotation JSON file")
    parser.add_argument('--tensor_folder', type=str, help="Folder containing existence and location tensors")
    parser.add_argument('--anchor_file', type=str, help="JSON file with anchor shapes (e.g., [[1,1],[2,1],[1,2]])")
    parser.add_argument('--threshold', type=float, default=0.65, help="Confidence threshold for filtering boxes")
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    main(args)

    # Collaborators (if any):
    # e.g., Wei-Lun Chao, chao209