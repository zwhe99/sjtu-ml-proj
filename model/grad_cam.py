
import torch
import torchvision
import torch.nn.functional as F
from .utils import preprocess, normalize, IMAGE_SHAPE, top_k_indices
from tqdm import tqdm
from .base_explainer import BaseExplainer
from .utils import preprocess, normalize


class GradCAM(BaseExplainer):

    def __init__(self, model, top_k):
        super(GradCAM, self).__init__(model, top_k)

        self.feature_map = None
        self.gradient = None

        if isinstance(self.model, torchvision.models.vgg.VGG):
            getattr(
                self.model, "features")[-1].register_forward_hook(self.forward_hook)
            getattr(
                self.model, "features")[-1].register_full_backward_hook(self.backward_hook)
        elif isinstance(self.model, torchvision.models.resnet.ResNet):
            getattr(self.model, "layer4").register_forward_hook(
                self.forward_hook)
            getattr(self.model, "layer4").register_full_backward_hook(
                self.backward_hook)
        else:
            raise ValueError(f"Only {torchvision.models.vgg.VGG} and {torchvision.models.resnet.ResNet} are allowed for GradCAM.")

    def explain(self, input_image):
        input_image = normalize(preprocess(input_image))
        target_ids = top_k_indices(self.model, input_image, self.top_k)

        result = torch.empty((self.top_k, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))

        for i, tgt_id in tqdm(enumerate(target_ids), total=self.top_k):
            result[i] = self.get_importance_values(input_image, tgt_id)

        return result, target_ids

    def forward_hook(self, module, input, output):
        self.feature_map = output.squeeze().detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].squeeze().detach()

    def get_importance_values(self, input_image, tgt_id):
        self.forward_backward(input_image, tgt_id)

        alpha = self.gradient.mean(dim=[1, 2])

        grad_cam = F.relu(torch.einsum('c,chw->hw', alpha, self.feature_map))
        grad_cam = F.interpolate(grad_cam.view(
            1, 1, *grad_cam.shape), mode="bicubic", size=IMAGE_SHAPE)

        grad_cam = (grad_cam - grad_cam.min()) / grad_cam.max()
        return grad_cam.squeeze()

    def forward_backward(self, input_image, tgt_id):
        self.model.zero_grad()

        # forward
        output = self.model(input_image.unsqueeze(0))

        # backward
        loss = torch.sum(
            output * F.one_hot(tgt_id, num_classes=output.shape[1]))
        loss.backward()
