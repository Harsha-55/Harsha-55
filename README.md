import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image

# Define the transform to preprocess images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a function to load and preprocess an image
def load_image(image_path, transform=None, max_size=None):
    image = Image.open(image_path)
    if max_size is not None:
        image.thumbnail((max_size, max_size), Image.ANTIALIAS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image.to(device)

# Define the VGG model to extract features
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = models.vgg19(pretrained=True).features[:37]

    def forward(self, x):
        return self.features(x)

# Define the content loss function
def content_loss(target, content):
    return torch.mean((target - content) ** 2)

# Define the style loss function
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def style_loss(target, style):
    _, c, h, w = target.size()
    target_gram = gram_matrix(target)
    style_gram = gram_matrix(style)
    return torch.mean((target_gram - style_gram) ** 2)

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the content and style images
content_image = load_image("content.jpg", transform, max_size=400)
style_image = load_image("style.jpg", transform, max_size=400)

# Initialize the generated image with the content image
generated_image = content_image.clone().requires_grad_(True).to(device)

# Load the VGG model and set it to evaluation mode
vgg = VGG().to(device).eval()

# Define the optimizer
optimizer = optim.Adam([generated_image], lr=0.01)

# Number of iterations and weights for content and style losses
num_epochs = 300
content_weight = 1
style_weight = 1e5

# Main training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    content_features = vgg(content_image)
    style_features = vgg(style_image)
    generated_features = vgg(generated_image)

    # Calculate losses
    loss_content = content_loss(generated_features, content_features)
    loss_style = style_loss(generated_features, style_features)
    total_loss = content_weight * loss_content + style_weight * loss_style

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    # Print the loss every 50 iterations
    if (epoch + 1) % 50 == 0:
        print(f"Iteration [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item()}")

# Save the final generated image
output_image = generated_image.squeeze(0).detach().cpu()
output_image = transforms.ToPILImage()(output_image)
output_image.save("output.jpg")
