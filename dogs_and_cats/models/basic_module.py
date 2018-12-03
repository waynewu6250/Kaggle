import torch as t
import time

class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))
    
    def load(self, path):
        self.load_state_dict(t.load(path))
    
    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.module_name + '_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
