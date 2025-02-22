from torch import nn


class DynamicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {}

    def define_layer_dynamically(self, name: str, module_class, *args, **kwargs):
        if not hasattr(self, name):
            self.add_module(name, module_class(*args, **kwargs))
        return getattr(self, name)

    def state_dict(self, destination=None, prefix='', keep_vars=False, **kwargs):
        state = super().state_dict(destination, prefix, keep_vars, **kwargs)
        for name, module in self.named_modules():
            if name and name not in state:
                state[f"{prefix}{name}"] = module.state_dict(destination, prefix + name + ".", keep_vars, **kwargs)
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for name, module_state in state_dict.items():
            if "." not in name:
                if not hasattr(self, name):
                    module_class = type(module_state)
                    self.add_module(name, module_class())
            getattr(self, name).load_state_dict(module_state, strict=strict)
        super().load_state_dict(state_dict, strict=False)