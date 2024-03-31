import torch
from PIL import Image
from model.lennon import LeNNon
from model.blazeface.blazeface import BlazeFace
from torchvision import transforms
from io import BytesIO

# lennon loading fix
import __main__
__main__.LeNNon=LeNNon



class SmileDetector:

    def initFaceDetectionModel(self):
        modelDir = "model/blazeface/"
        front_net = BlazeFace().to(self.device)
        front_net.load_weights(modelDir + "blazeface.pth")
        front_net.load_anchors(modelDir + "anchors.npy")
        back_net = BlazeFace(back_model=True).to(self.device)
        back_net.load_weights(modelDir + "blazefaceback.pth")
        back_net.load_anchors(modelDir + "anchorsback.npy")

        # Optionally change the thresholds:
        front_net.min_score_thresh = 0.75
        front_net.min_suppression_threshold = 0.3

        self.faceDetectionForwardModel = front_net
        self.faceDetectionBackModel = back_net
    def initSmileModel(self):
        # Load the model an pass it to the proper device
        modelPath = 'model/LeNNon-Smile-Detector.pt'
        smileModel = torch.load(modelPath)
        smileModel = smileModel.to(self.device)
        smileModel.eval()

        # This `transform` object will transform our test images into proper tensors
        transform = transforms.Compose([
            transforms.Resize((100, 100)),  # Resize the image to 100x100
            transforms.ToTensor(),
        ])

        self.smileModel = smileModel
        self.smileTransform = transform

    def __init__(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.initSmileModel()
        self.initFaceDetectionModel()

    def smileCheck(self, image_bytes: bytes):
        # Open and preprocess he image
        image_io = BytesIO(image_bytes)
        image = Image.open(image_io)
        width, height = image.size   # Get dimensions
        imageCrop = transforms.CenterCrop((min(width,height),min(width,height)))

        tensor = self.smileTransform(imageCrop(image))
        tensor = tensor.to(self.device)

        # forward pass trough the model
        with torch.no_grad():

            outputs = self.smileModel(tensor)

        # Get the class prediction
        _, predicted = torch.max(outputs.data, 1)

        return predicted.item() > 0

    def findFaces(self, image_bytes: bytes):
        image_io = BytesIO(image_bytes)
        image = Image.open(image_io)
        width, height = image.size   # Get dimensions
        baseDim = min(width,height)
        transform = transforms.Compose([
            transforms.CenterCrop((min(width,height),min(width,height))),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            lambda x: x*255 # to be compatible with the model
        ])

        detections = self.faceDetectionBackModel.predict_on_image(transform(image))
        res = []
        for i in range(detections.shape[0]):
            faceMarkers = detections[i]

            ymin = faceMarkers[0].item() * baseDim # + (width - height)/2
            xmin = faceMarkers[1].item() * baseDim + (width - height)/2
            ymax = faceMarkers[2].item() * baseDim # + (width - height)/2
            xmax = faceMarkers[3].item() * baseDim + (width - height)/2
            res.append((ymin,xmin,ymax,xmax))
        return res
