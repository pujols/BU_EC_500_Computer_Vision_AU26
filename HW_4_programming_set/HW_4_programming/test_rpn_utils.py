import unittest
import numpy as np
from PIL import Image
import os

from main import (
    resize_image_and_boxes,
    compute_patch_grid,
    boxes_overlap,
    generate_gt_tensors,
    compute_iou,
    non_max_suppression,
    change_coordinates,
    PATCH_SIZE,
    GRID_NUMBER,
    TARGET_SIZE,
)

class TestRPNUtils(unittest.TestCase):

    def setUp(self):
        self.sample_image = Image.new("RGB", (400, 300), color="white")
        self.image_path = "temp_sample.jpg"
        self.sample_image.save(self.image_path)

    def tearDown(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)

    # ---------------- resize_image_and_boxes ----------------
    def test_resize_image_and_boxes_simple(self):
        boxes = [[40, 30, 100, 50]]
        image, resized = resize_image_and_boxes(self.image_path, boxes)
        self.assertEqual(image.size, (TARGET_SIZE, TARGET_SIZE))
        self.assertAlmostEqual(resized[0][0], 40 * TARGET_SIZE / 400)

    def test_resize_image_and_boxes_empty(self):
        boxes = []
        image, resized = resize_image_and_boxes(self.image_path, boxes)
        self.assertEqual(len(resized), 0)

    def test_resize_image_and_boxes_multiple_boxes(self):
        boxes = [[10, 10, 20, 20], [50, 50, 100, 100]]
        _, resized = resize_image_and_boxes(self.image_path, boxes)
        self.assertEqual(len(resized), 2)
        self.assertTrue(all(len(b) == 4 for b in resized))

    # ---------------- compute_patch_grid ----------------
    def test_patch_grid_shape(self):
        grid = compute_patch_grid()
        self.assertEqual(len(grid), GRID_NUMBER * GRID_NUMBER)

    def test_patch_grid_content(self):
        grid = compute_patch_grid()
        i, j, x_center, y_center, w, h  = grid[0]
        self.assertEqual(w - x_center, PATCH_SIZE/2)
        self.assertEqual(h - y_center, PATCH_SIZE/2)

    def test_patch_grid_last_patch(self):
        grid = compute_patch_grid()
        i, j, x_center, y_center, w, h  = grid[-1]
        self.assertLessEqual(w, TARGET_SIZE)
        self.assertLessEqual(h, TARGET_SIZE)

    # ---------------- boxes_overlap ----------------
    def test_overlap_true(self):
        self.assertTrue(boxes_overlap([10, 10, 30, 30], [20, 20, 30, 30]))

    def test_overlap_false(self):
        self.assertFalse(boxes_overlap([0, 0, 10, 10], [20, 20, 10, 10]))

    def test_overlap_edge_touching(self):
        self.assertTrue(boxes_overlap([0, 0, 10, 10], [5, 0, 10, 10]))

    # ---------------- generate_gt_tensors ----------------
    def test_generate_gt_tensor_shapes(self):
        boxes = [[20, 20, 10, 10]]
        existence, location = generate_gt_tensors(boxes)
        self.assertEqual(existence.shape, (GRID_NUMBER, GRID_NUMBER, 2))
        self.assertEqual(location.shape, (GRID_NUMBER, GRID_NUMBER, 4))

    def test_generate_gt_tensor_existence_sum(self):
        boxes = [[20, 20, 10, 10]]
        existence, _ = generate_gt_tensors(boxes)
        total_cells = GRID_NUMBER * GRID_NUMBER
        self.assertEqual(np.sum(existence), total_cells)

    def test_generate_gt_tensor_positive_patch(self):
        boxes = [[20, 20, 10, 10]]
        existence, location = generate_gt_tensors(boxes)
        self.assertTrue(np.any(existence[:, :, 0] == 1))  # object present
        self.assertTrue(np.any(location[:, :, 2] > 0))    # non-zero width

    # ---------------- compute_iou ----------------
    def test_iou_overlap(self):
        iou = compute_iou([0, 0, 100, 100], [50, 50, 100, 100])
        self.assertGreater(iou, 0.0)
        self.assertLessEqual(iou, 1.0)

    def test_iou_no_overlap(self):
        iou = compute_iou([0, 0, 50, 50], [100, 100, 50, 50])
        self.assertEqual(iou, 0.0)

    def test_iou_exact_match(self):
        iou = compute_iou([10, 10, 30, 30], [10, 10, 30, 30])
        self.assertEqual(iou, 1.0)

    # ---------------- non_max_suppression ----------------
    def test_nms_overlap_suppression(self):
        boxes = [
            (100, 100, 50, 50, 0.9),
            (105, 105, 50, 50, 0.8),
            (300, 300, 50, 50, 0.7)
        ]
        result = non_max_suppression(boxes, threshold=0.5)
        self.assertEqual(len(result), 2)
        self.assertIn((100, 100, 50, 50, 0.9), result)

    def test_nms_no_suppression(self):
        boxes = [
            (10, 10, 20, 20, 0.5),
            (100, 100, 20, 20, 0.6)
        ]
        result = non_max_suppression(boxes, threshold=0.1)
        self.assertEqual(len(result), 2)

    def test_nms_empty_input(self):
        result = non_max_suppression([], threshold=0.5)
        self.assertEqual(result, [])

    # ---------------- change_coordinates ----------------
    def test_change_coordinates_simple(self):
        image_dict = {'width': 100, 'height': 100}
        bbox = [10, 20, 30, 40]  # [x_min, y_min, w, h]
        result = change_coordinates(bbox, image_dict)
        expected = [25, 60, 30, 40]  # [y_min, H - x_min - h, w, h]
        self.assertEqual(result, expected)

    def test_change_coordinates_top_left_corner(self):
        image_dict = {'width': 50, 'height': 80}
        bbox = [0, 0, 10, 20]
        result = change_coordinates(bbox, image_dict)
        expected = [5, 70, 10, 20]
        self.assertEqual(result, expected)

    def test_change_coordinates_bottom_edge(self):
        image_dict = {'width': 60, 'height': 50}
        bbox = [40, 20, 10, 10]
        result = change_coordinates(bbox, image_dict)
        expected = [45, 25, 10, 10]  # [y_min, H - x_min - h, w, h]
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
