import torch as t
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os

#Our Defined modules
from config import opt
from data import DogCat
import models
from utils import Visualizer

def train(**kwargs):

    #1. Load parameter and vis
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    #2. Load data
    train_data = DogCat(opt.train_data_root, train = True)
    val_data = DogCat(opt.test_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False)

    #3. Load model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    
    #4. Load utils
    vis = Visualizer(opt.env)

    #################################################################
    
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = Adam(model.parameters(), lr=lr, weight_decay = opt.weight_decay)

    for epoch in range(opt.epoch):
        
        ##=========================================================##
        ##                         train                           ##
        ##=========================================================##
        running_loss = []
        for i, (x_batch, y_batch) in enumerate(train_dataloader):
            data = Variable(x_batch)
            label = Variable(y_batch)
            
            optimizer.zero_grad()
            score = model(data)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.data[0])
            if i % 2000 == 1999: # print every 2000 mini-batches
                vis.plot('loss',loss.data[0])
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, sum(running_loss) / 2000))
                running_loss = 0.0

                # debugging
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save()
        ##=========================================================##
        ##                        validate                         ##
        ##=========================================================##
        model.val()
        
        validate_loss = []
        for i, (x_batch, y_batch) in enumerate(val_dataloader):
            data = Variable(x_batch, volatile=True)
            label = Variable(y_batch.long(), volatile=True)

            score = model(data)
            loss = criterion(score, label)
            validate_loss.append(loss.data[0])
            vis.plot('loss',loss.data[0])

        vis.log("epoch:{epoch},lr:{lr},loss:{loss}". \
                format(
                        epoch = epoch,
                        lr = lr,
                        loss = sum(running_loss) / 2000
                ))

        #############################################################
        if loss.data[0] > running_loss[-1]:
            lr *= opt.lr_decay
            for param in optimizer.param_groups:
                param["lr"] = lr


