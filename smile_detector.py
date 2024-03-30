import torch
from PIL import Image
from model.lennon import LeNNon
from torchvision import transforms
from io import BytesIO

# lennon loading fix
import __main__
__main__.LeNNon=LeNNon


class SmileDetector:
    def __init__(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model an pass it to the proper device
        modelPath = 'model/LeNNon-Smile-Detector.pt'
        model = torch.load(modelPath)
        model = model.to(device)
        model.eval()

        # This `transform` object will transform our test images into proper tensors
        transform = transforms.Compose([
            transforms.Resize((100, 100)),  # Resize the image to 100x100
            transforms.ToTensor(),
        ])

        self.device = device
        self.model = model
        self.transform = transform

    def smileCheck(self, image_bytes: bytes):
        # Open and preprocess he image
        image_io = BytesIO(image_bytes)
        image = Image.open(image_io)
        tensor = self.transform(image)
        tensor = tensor.to(self.device)

        # forward pass trough the model
        with torch.no_grad():

            outputs = self.model(tensor)

        # Get the class prediction
        _, predicted = torch.max(outputs.data, 1)

        return predicted.item() > 0