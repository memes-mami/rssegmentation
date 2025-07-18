from .base_dataset import BaseDataset
from PIL import Image
import os
import numpy as np

class LoveDA(BaseDataset):
    def __init__(self, data_root='data/vaihingen', mode='train', transform=None,
                 img_dir='images_png', mask_dir='masks_png', img_suffix='.png', mask_suffix='.png', **kwargs):
        super(LoveDA, self).__init__(transform)

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.mode = mode

        if mode == "train":
            self.data_root = data_root + "/Train"
        elif mode == "val":
            self.data_root = data_root + "/Val"
        elif mode == "test":
            self.data_root = data_root + "/Test"

        self.file_paths = self.get_path(self.data_root, img_dir, mask_dir)

        # RGB
        self.color_map = {
            'building': np.array([255, 0, 0]),       # label 0
            'road': np.array([255, 255, 0]),         # label 1
            'water': np.array([0, 0, 255]),          # label 2
            'barren': np.array([159, 129, 183]),     # label 3
            'forest': np.array([0, 255, 0]),         # label 4
            'agricultural': np.array([255, 195, 128]),  # label 5
            'background': np.array([255, 255, 255])  # label 6
        }

        self.num_classes = 7

    def get_path(self, data_root, img_dir, mask_dir):
        img_ids = []

        # Urban
        urban_img_path = os.path.join(data_root, 'Urban', img_dir)
        urban_mask_path = os.path.join(data_root, 'Urban', mask_dir)
        urban_img_filename_list = os.listdir(urban_img_path)

        if self.mode != 'test' and os.path.exists(urban_mask_path):
            urban_mask_filename_list = os.listdir(urban_mask_path)
            if len(urban_img_filename_list) != len(urban_mask_filename_list):
                print(f"[Warning] Mismatch in Urban images and masks: {len(urban_img_filename_list)} vs {len(urban_mask_filename_list)}")
        elif self.mode != 'test':
            print(f"[Info] Urban mask directory not found: {urban_mask_path}")

        urban_img_ids = [(str(id.split('.')[0]), 'Urban') for id in urban_img_filename_list]
        img_ids += urban_img_ids

        # Rural
        rural_img_path = os.path.join(data_root, 'Rural', img_dir)
        rural_mask_path = os.path.join(data_root, 'Rural', mask_dir)
        rural_img_filename_list = os.listdir(rural_img_path)

        if self.mode != 'test' and os.path.exists(rural_mask_path):
            rural_mask_filename_list = os.listdir(rural_mask_path)
            if len(rural_img_filename_list) != len(rural_mask_filename_list):
                print(f"[Warning] Mismatch in Rural images and masks: {len(rural_img_filename_list)} vs {len(rural_mask_filename_list)}")
        elif self.mode != 'test':
            print(f"[Info] Rural mask directory not found: {rural_mask_path}")

        rural_img_ids = [(str(id.split('.')[0]), 'Rural') for id in rural_img_filename_list]
        img_ids += rural_img_ids

        return img_ids

    def load_img_and_mask(self, index):
        img_id, img_type = self.file_paths[index]
        img_name = os.path.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        img = Image.open(img_name).convert('RGB')

        if self.mode == 'test':
            mask = np.array(img)
            mask = Image.fromarray(mask).convert('L')
            return [img, mask, img_id]

        mask_name = os.path.join(self.data_root, img_type, self.mask_dir, img_id + self.mask_suffix)

        if not os.path.exists(mask_name):
            print(f"[Warning] Mask not found for {img_id} at {mask_name}, using blank mask.")
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8)).convert('L')
        else:
            mask = Image.open(mask_name).convert('L')
            np_mask = np.array(mask)
            np_mask[np_mask == 0] = 8
            np_mask -= 1
            mask = Image.fromarray(np_mask).convert('L')

        return [img, mask, img_id]
