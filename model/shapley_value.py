import torch
from einops import rearrange
from .utils import preprocess, normalize, top_k_indices, IMAGE_SHAPE
from tqdm import tqdm
import random
import math

BASE_VALUE = 128 / 255

class ShapleyValue:

    def __init__(self, model, max_context, patch_side_len, top_k):
        self.model = model
        self.max_context = max_context
        self.patch_side_len = patch_side_len
        self.top_k = top_k
        self.grid_shape = (IMAGE_SHAPE[0] // self.patch_side_len, IMAGE_SHAPE[0] // self.patch_side_len)
        self.num_patches = self.grid_shape[0] * self.grid_shape[1]
    
    def explain(self, input_image):
        input_image = preprocess(input_image)
        target_ids = top_k_indices(self.model, normalize(input_image), self.top_k)
        
        result = torch.empty((self.top_k, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))

        for i, tgt_id in tqdm(enumerate(target_ids), total=self.top_k):
            result[i] = self.get_shapley_values(input_image, tgt_id)
        
        return result, target_ids
    
    def get_shapley_values(self, input_image, tgt_id):
        input_grid = rearrange(input_image, 'c (h p0) (w p1) -> c p0 p1 (h w)', p0=self.patch_side_len, p1=self.patch_side_len)

        # compute shapley_value matrix (patch level)
        shapley_values = torch.zeros((self.grid_shape))
        for i in range(self.num_patches):
            # for each patch i, compute shapley value i
            
            # coordinates or cur patch
            r = i // self.grid_shape[1]
            c = i % self.grid_shape[1]

            shapley_value_i = 0
            for _ in range(self.max_context):

                # sample a mask
                threshold = random.random()
                mask = torch.rand((self.num_patches, )) < threshold
                
                # compute context
                context_mask = mask.clone()
                context_mask[i] = True                                                 # must mask patch i
                context_grid = torch.masked_fill(input_grid, context_mask, BASE_VALUE) # mask with base value
                context = rearrange(
                        context_grid, 
                        'c p0 p1 (h w) -> c (h p0) (w p1)', 
                        h=self.grid_shape[0], w=self.grid_shape[1]
                    )                                                                  # reconstrut grid to image
                context = normalize(context)                                           # normalize here

                # compute context with patch
                context_with_i_mask = mask.clone()
                context_with_i_mask[i] = False                                         # must not mask patch i
                context_with_i_grid = torch.masked_fill(input_grid, context_with_i_mask, BASE_VALUE)
                context_with_i = rearrange(
                        context_with_i_grid, 
                        'c p0 p1 (h w) -> c (h p0) (w p1)', 
                        h=self.grid_shape[0], w=self.grid_shape[1]
                    )
                context_with_i = normalize(context_with_i)
                
                
                shapley_value_i += self.value(context_with_i, tgt_id) - self.value(context, tgt_id)

            shapley_value_i /= self.max_context # we need average value over different contexts
            shapley_values[r, c] = shapley_value_i
        
        # Shapley_values are patch level. To ensure consistency with the size of the image, it is now scaled to pixel level.
        shapley_values_grid = torch.empty((self.patch_side_len, self.patch_side_len, self.num_patches))
        for i in range(self.num_patches):
            r = i // self.grid_shape[1]
            c = i % self.grid_shape[1]
            select = torch.full((self.num_patches,), False) 
            select[i] = True
            shapley_values_grid = torch.masked_fill(shapley_values_grid, select, shapley_values[r, c])
        shapley_values_img = rearrange(shapley_values_grid, 'p0 p1 (h w) -> (h p0) (w p1)', h=self.grid_shape[0], w=self.grid_shape[1])
        return shapley_values_img


    def value(self, x, tgt_id):
        x = x.unsqueeze(0) 

        with torch.no_grad():
            output = self.model(x)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        prob = probabilities[tgt_id]
        return math.log(prob / (1 - prob))






