import torch 
from torch.utils.data import Dataset
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

class SyntheticCLEVRDataset(Dataset):
    # |occlusion_tolerance| is the minimum proportion of the object that should be visible.
    # |max_num_objects| is the maximum number of objects that should be present in the final image
    def __init__(self, filename, max_num_objects, occlusion_tolerance):
        with open(filename, "rb") as file:
            dict = pk.load(file)
            # dimensions - background: n1 x c1 x h1 x w1, image: n2 x c1 x h2 x w2, 0-1 mask: n2 x h2 x w2
            # all components are assumed to be python lists
            backgrounds, images, masks = dict["background"], dict["image"], dict["masks"]
        
        self.backgrounds = torch.tensor(backgrounds, dtype=torch.long)
        self.images = torch.tensor(images, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.long)
        self.max_num_objects = max_num_objects
        self.occlusion_tolerance = occlusion_tolerance
    
    def __getitem__(self, unused_index):
        num_backgrounds, channel_canvas, height_canvas, width_canvas = self.backgrounds.shape
        num_images, _, height_images, width_images = self.images.shape

        num_objects = np.random.randint(1,self.max_num_objects+1)

        rnd_images_idxes = np.random.choice(np.arange(num_images), num_objects, replace=True)
        canvas = -1*torch.ones((channel_canvas, height_canvas, width_canvas), dtype=torch.long)
        mask_canvas = torch.zeros((height_canvas, width_canvas), dtype=torch.long) #for computational purposes only
        
        output_mask_list = []
        output_bounding_box_list = []
        # Number of attempts we make to place an image randomly
        max_tries = 10 
        for img_idx in rnd_images_idxes:
            curr_img = self.images[img_idx]
            curr_mask = self.masks[img_idx]
            
            for tries in range(max_tries):
                # |curr_tl_x| and |curr_tl_y| denote the top left coordinate of the prospective location 
                # to insert the image
                curr_tl_x = np.random.randint(0, height_canvas-height_images)
                curr_tl_y = np.random.randint(0, width_canvas-width_images)

                # Extract the local mask-canvas for |curr_tl_x| and |curr_tl_y|
                local_mask_canvas = mask_canvas[curr_tl_x:curr_tl_x+height_images, curr_tl_y:curr_tl_y+width_images]
                
                # Remove locations from the mask which already have an image
                curr_mask_diff = torch.sub(curr_mask,local_mask_canvas)
                curr_mask_diff[curr_mask_diff < 0] = 0

                # If a substantial portion of the current image can be placed
                if torch.sum(curr_mask_diff)/torch.sum(curr_mask) >= self.occlusion_tolerance:
                    local_mask_canvas.add_(curr_mask_diff)
                    masked_img = curr_img*curr_mask_diff

                    local_canvas = canvas[:, curr_tl_x:curr_tl_x+height_images, curr_tl_y:curr_tl_y+width_images]
                    local_canvas[masked_img>0] = 0
                    local_canvas = torch.add(local_canvas, masked_img)

                    output_mask_list.append(curr_mask_diff)
                    output_bounding_box_list.append([curr_tl_x, curr_tl_y, curr_tl_x + height_images-1, curr_tl_y+width_images-1])
                    break

        # Choose a background
        background_idx = np.random.randint(0, num_backgrounds)
        # Add the background
        canvas[canvas == -1] = self.backgrounds[background_idx][canvas == -1]

        return [canvas, torch.tensor(output_bounding_box_list), torch.stack(output_mask_list)]