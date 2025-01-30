import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from vgg_pre import ModelPart1, ModelPart2, ModelPart3, ModelPart4


# grpc 
from concurrent import futures
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

class Exchanger(tensor_pb2_grpc.ExchangerServicer):
    
    def ExchangeTensor(self, request, context):
        tensor = torch.tensor(request.tensor).view(*request.size)
        if(request.id==1):
            tensor = model_1(tensor)
        elif(request.id==2):
            tensor = model_2(tensor)
        elif(request.id==3):
            tensor = model_3(tensor)
        elif(request.id==4):
            tensor = model_4(tensor)
        return tensor_pb2.Tensor(id=request.id+1, size=list(tensor.size()), tensor=tensor.view(-1).tolist())
    
def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tensor_pb2_grpc.add_ExchangerServicer_to_server(Exchanger(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()