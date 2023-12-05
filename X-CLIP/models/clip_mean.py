import clip
import torch
from PIL import Image
from clip_mean import clip

class CLIPBenchmark(torch.nn.Module):
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(CLIPBenchmark, self).__init__()
        self.model, self.preprocess = clip.load(model, device=device)
        self.device = device

    def forward(self, text_labels, images):
        # Preprocess text labels
        text_inputs = text_labels.to(self.device)

        # Preprocess images
        image_inputs = torch.stack([self.preprocess(Image.fromarray(img.cpu().numpy())) for img in images]).to(self.device)

        # Forward pass through CLIP model
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            text_features = self.model.encode_text(text_inputs)


        mean_image_features = image_features.mean(dim=0)
        # Calculate similarity

        logits = (mean_image_features @ text_features.T)

        return logits
