import numpy as np
import os
#import torch
import matplotlib.pyplot as plt
import requests
from PIL import Image
from torchvision import models, transforms
from generate_vat import *

torch.manual_seed(123)

class Get_Feature_Maps():
    def __init__(self,data_dir='./data/',model='resnet50',save_features='False',save_file='features',max_pooling='False',generate_vat='False',resize=224):

        self.data_dir = data_dir
        self.model = model
        self.save_features = save_features
        self.save_file = save_file
        self.max_pooling = max_pooling
        self.generate_vat = generate_vat
        self.resize = resize
        self.num_images = len(os.listdir(data_dir))

        # Let's get our class labels.
        LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
        response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
        labels = {int(key): value for key, value in response.json().items()}

        self.trans = transforms.Compose([ transforms.Resize((resize,resize)),
                                     transforms.ToTensor(),
                                     transforms.Normalize( [0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225] ) ])

        if(self.model.lower() == 'alexnet'):

            self.model1 = models.alexnet(pretrained=True)

        elif(self.model.lower() == 'google'):

            self.model1 = models.googlenet(pretrained=True)

        elif(self.model.lower() == 'resnet18'):

            self.model1 = models.resnet18(pretrained=True)

        elif(self.model.lower() == 'resnet50'):

            self.model1 = models.resnet50(pretrained=True)

        elif(self.model.lower() == 'resnet101'):

            self.model1 = models.resnet101(pretrained=True)

        elif(self.model.lower() == 'vgg'):

            self.model1 = models.vgg16(pretrained=True)

        else:

            print('\nInvalid model.... Exiting!\n')
            exit()

        self.new_model = torch.nn.Sequential(*list(self.model1.children())[:-2])


    def evaluate(self):
        # Evaluate
        self.model1.eval()
        self.new_model.eval()

        array = torch.randn(3,224,224)
        new_preds = self.new_model(array.unsqueeze(0))
        nparr = new_preds.detach().numpy()
        features = []
        names = []
        files = sorted(os.listdir(self.data_dir))

        for i,pic in enumerate(files):
            try:
                img = Image.open(self.data_dir+pic)
                img = self.trans(img)
                #preds = self.model1(img.unsqueeze(0))
                #print(labels[preds.data.numpy().argmax()])
                new_preds = self.new_model(img.unsqueeze(0))
                nparr = new_preds.detach().numpy()
                num_features = nparr.shape[1]

                if(str(self.max_pooling) == 'True'):
                    vect = list()
                    for max in nparr[0]:
                        vect.append(max.max())
                else:
                    vect = nparr.ravel()

                print(pic)
                names.append(pic)
                features.append(vect)

            except:
                print("Error opening: ",self.data_dir,pic)


        if(str(self.save_features) == 'True'):
            np.save(self.save_file,features)

        if(str(self.generate_vat) == 'True'):
            #Use VAT_Image object to display differential matrix
            VAT = VAT_Image(features,self.data_dir,self.model,self.max_pooling)
            VAT.show()

        return features

