from torch import nn


class DynamicModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = {}

    def define_layer_dynamically(self, name: str, module_class, *args, **kwargs):
        if not hasattr(self, name):
            self.add_module(name, module_class(*args, **kwargs))
        return getattr(self, name)
