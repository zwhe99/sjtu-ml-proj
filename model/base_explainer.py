import torch
from tqdm import tqdm
from .utils import preprocess, top_k_indices, normalize, IMAGE_SHAPE

class BaseExplainer:
    def __init__(self, model, top_k):
        self.model = model
        self.model.eval()
        
        self.top_k = top_k
    
    def explain(self, input_image):
        input_image = preprocess(input_image)
        target_ids = top_k_indices(self.model, normalize(input_image), self.top_k)
        
        result = torch.empty((self.top_k, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))

        for i, tgt_id in tqdm(enumerate(target_ids), total=self.top_k):
            result[i] = self.get_importance_values(input_image, tgt_id)
        
        return result, target_ids
    
    def get_importance_values(self, input_image, tgt_id): 
        raise NotImplementedError("The subclass should override BaseExplainer.get_importance_values method.")