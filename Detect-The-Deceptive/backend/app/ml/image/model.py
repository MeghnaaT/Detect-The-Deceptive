import torch
import torchvision.models as models

class EfficientNetWithHooks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights="DEFAULT")
        self.model.classifier[1] = torch.nn.Linear(1280, 2)

        self.gradients = None
        self.activations = None

        target_layer = self.model.features[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)

def load_model():
    model = EfficientNetWithHooks()
    model.eval()
    return model
