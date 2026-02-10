# Image model definition
import torch
import torchvision.models as models

class EfficientNetWithHooks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights="DEFAULT")
        self.model.classifier[1] = torch.nn.Linear(1280,2)

        self.gradients = None
       
        last_conv = self.model.features[-1]
        last_conv.register_full_backward_hook(self.save_gradients)

    def save_gradients(self,module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self,x):
        return self.model(x)
    
def load_model(): # Will load PyTorch model
    model = EfficientNetWithHooks()
     # Loads the EfficientNet-B0 architecture from torchvision.models

#Modify the classifier layer

    model.eval()
    return  model # modified  odel, ready for interference on a binary task classification task.

#Summary
#- Loads EfficientNet-B0 pretrained on ImageNet.
#- Replaces the final classification layer to output 2 classes instead of 1000.
#- Sets the model to evaluation mode.
#- Returns the model for inference.

# Example:
#model = load_model()
# print(model)
# Suppose you have a preprocessed image tensor:
# shape: (1, 3, 224, 224)
# output = model(image_tensor)
# print(output)  # shape: (1, 2) → logits for 2 classes