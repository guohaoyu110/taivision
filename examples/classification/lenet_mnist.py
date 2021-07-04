import taicv
import torch
import gluoncv
from tqdm import tqdm
import torch.nn.functional as F
from taicv.data import transforms


data_path = '../../data'
num_epoch = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])


train_dataset = taicv.data.MNIST(root=data_path,
                             train=True, 
                             transform=train_transforms, 
                             target_transform=None,
                             download=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64,
                                           shuffle=False)  
num_classes = len(train_dataset.classes)
model = taicv.model_zoo.lenet(num_classes=num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)

for epoch in range(num_epoch):
    for batch_idx, (img, target) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        output = model(img)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('epoch {},batch_idx {},loss {}'.format(epoch,batch_idx,loss.item()))
        break
    break
    