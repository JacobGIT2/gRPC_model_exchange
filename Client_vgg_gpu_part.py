import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from vgg_pre import ModelPart1, ModelPart2, ModelPart3, ModelPart4


# grpc 
import logging
import grpc
import tensor_pb2
import tensor_pb2_grpc
#

if __name__ == '__main__':

##########################
### SETTINGS
##########################

    seed=42
    torch.manual_seed(seed)

    # Architecture
    NUM_CLASSES = 100
    BATCH_SIZE = 1
    DEVICE = torch.device('cpu')

##########################
### CIFAR-100 Dataset
##########################

    test_dataset = datasets.CIFAR100(root='data', 
                                    train=False, 
                                    transform=transforms.ToTensor())


#####################################################
### Data Loaders
#####################################################


    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=BATCH_SIZE,
                            # num_workers=12,
                            shuffle=False)

#####################################################
    # Create empty model partitions
    model_1 = ModelPart1()
    model_2 = ModelPart2()
    model_3 = ModelPart3()
    model_4 = ModelPart4(num_classes=NUM_CLASSES)


    # Load state_dicts and set to eval mode and device
    model_1.load_state_dict(torch.load('../model_vgg/vgg_part1.pth',map_location='cpu'))
    model_1.eval().to(DEVICE)
    model_2.load_state_dict(torch.load('../model_vgg/vgg_part2.pth',map_location='cpu'))
    model_2.eval().to(DEVICE)
    model_3.load_state_dict(torch.load('../model_vgg/vgg_part3.pth',map_location='cpu'))
    model_3.eval().to(DEVICE)
    model_4.load_state_dict(torch.load('../model_vgg/vgg_part4.pth',map_location='cpu'))
    model_4.eval().to(DEVICE)


# ###############
# #evaluation
# ###############
    # Interface
    address = input("input the server's ipv4 address: ")
    signs = int(input("input the execution integer from 0(0000)-15(1111), from left to right: "))
    sign4 = signs & 1
    sign3 = (signs >> 1) & 1
    sign2 = (signs >> 2) & 1
    sign1 = (signs >> 3) & 1
    #
    with torch.no_grad(), grpc.insecure_channel(address + ':50051') as channel:
        # grpc
        stub = tensor_pb2_grpc.ExchangerStub(channel)
        #
        
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(test_loader):

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            ##
            if(not sign1):
                x = model_1(features)
            else:
                x = stub.ExchangeTensor(tensor_pb2.Tensor(id=1, size=list(features.size()), tensor=features.view(-1).tolist()))
                x = torch.tensor(x.tensor).view(*x.size)
            if(not sign2):
                x = model_2(x)
            else:
                x = stub.ExchangeTensor(tensor_pb2.Tensor(id=2, size=list(x.size()), tensor=x.view(-1).tolist()))
                x = torch.tensor(x.tensor).view(*x.size)
            if(not sign3):
                x = model_3(x)
            else:
                x = stub.ExchangeTensor(tensor_pb2.Tensor(id=3, size=list(x.size()), tensor=x.view(-1).tolist()))
                x = torch.tensor(x.tensor).view(*x.size)
            if(not sign4):
                out = model_4(x)
            else:
                out = stub.ExchangeTensor(tensor_pb2.Tensor(id=4, size=list(x.size()), tensor=x.view(-1).tolist()))
                out = torch.tensor(out.tensor).view(*out.size)
            ##
            _, predicted_labels = torch.max(out, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

            if (i==30):
                print('Test accuracy: %.2f%%' % (correct_pred.float()/num_examples * 100))
                break