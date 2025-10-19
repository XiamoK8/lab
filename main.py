import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from model import CeiT

############  Hyper Parameters   ############
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#############  DATASET   ################
train_dataset = datasets.MNIST(root='../data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='../data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False)

###############   Model   ##################
model = CeiT(
    img_size=28,
    patch_size=4,
    in_channels=1,
    num_classes=10,
    embed_dim=192,
    depth=6,
    num_heads=4,
    mlp_ratio=4.,
    qkv_bias=True,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1
)
if args.cuda:
    model.cuda()
print(model)

################   Loss   #################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

################   Training   #############
def train(epoch):
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        if args.cuda:
            images = images.cuda()
            labels = labels.cuda()

        # forward
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()
        if (i+1) % args.log_interval == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   %(epoch+1, args.epochs, i+1, len(train_dataset)//args.batch_size, loss.item()))

def test():
    model.eval()
    correct_train = 0
    correct_test = 0

    with torch.no_grad():
        # training data test
        for images, labels in train_loader:
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()

        # testing data test
        for images, labels in test_loader:
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            _, predicted = outputs.max(1)
            correct_test += predicted.eq(labels).sum().item()

    trainacc = correct_train * 1.0 / len(train_dataset)
    testacc = correct_test * 1.0 / len(test_dataset)

    # logging
    print('Accuracy on training data is: %f . Accuracy on testing data is: %f. ' % (trainacc, testacc))

##################   Main   ##################
for epoch in range(args.epochs):
    train(epoch)
    test()
torch.save(model.state_dict(), '../model.pkl')