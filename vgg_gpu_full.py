import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from vgg_pre import vgg19_bn

##########################
### MODEL
##########################


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
    model = vgg19_bn()
    model.load_state_dict(torch.load('../model_vgg/vgg_full.pth',map_location='cpu'))
    model.eval().to(DEVICE)


# ###############
# #evaluation
# ###############

    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(test_loader):

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            out = model(features)

            _, predicted_labels = torch.max(out, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
 
            if (i==30):
                print('Test accuracy: %.2f%%' % (correct_pred.float()/num_examples * 100))
                break 