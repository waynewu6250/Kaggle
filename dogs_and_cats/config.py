import torch as t

class DefaultConfig:
    env = 'default'
    model = 'AlexNet'

    train_data_root = './data/train/'
    test_data_root = './data/test1'
    load_model_path = 'checkpoints/model.pth'

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    batch_size = 128
    epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

    # Make use of modify parameter directly
    def parse (self, kwargs):
        #kwargs is a dictionary to store every parameter that you would like to change
        #like {'lr' = 0.01, 'batch_size' = 64}
        for k, v in kwargs.items():
            if not hasattr(self, k):
                print("Warning: opt has no attribute %s" % k)
            setattr(self,k,v)
        
        #Print variables
        for k,v in self.__class__.__dict__.items():
            if not k.startswith("__"):
                print(k, getattr(self, k))

opt = DefaultConfig()