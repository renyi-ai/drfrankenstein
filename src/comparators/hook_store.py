import torch


def mean_accumulator(values):
    torch.cat(values, dim=0).mean()


class HookStore:
    def __init__(self) -> None:
        super().__init__()
        self.hooks = {}
        self.cache = {}
        self.active_model_name = ""

    def register_activation_saver(self, layer, key, model_name=None):
        if model_name is None:
            model_name = key

        def save_activations(module, m_in, m_out):
            if key not in self.cache:
                self.cache[key] = []
            if model_name == self.active_model_name:
                self.cache[key].append(m_out.detach().cpu())

        hook = layer.register_forward_hook(save_activations)
        self.hooks[key] = hook

    def register_gradient_saver(self, layer, key, model_name=None):
        if model_name is None:
            model_name = key

        def save_gradients(module, m_in, m_out):
            if key not in self.cache:
                self.cache[key] = []
            if model_name == self.active_model_name:
                self.cache[key].append(m_out[0].detach().cpu())

        hook = layer.register_backward_hook(save_gradients)
        self.hooks[key] = hook

    def get_and_clear_cache(self, key):
        cache = self.cache[key]
        self.delete_cache_entry(key)
        return cache

    def delete_hook_and_cache_entry(self, key):
        self.delete_hook(key)
        self.delete_cache_entry(key)

    def clear_store(self):
        self.clear_hooks()
        self.clear_cache()

    def clear_hooks(self):
        keys = list(self.hooks.keys())
        for key in keys:
            self.delete_hook(key)

    def delete_hook(self, key):
        self.hooks[key].remove()
        del self.hooks[key]

    def clear_cache(self):
        keys = list(self.hooks.keys())
        for key in keys:
            self.delete_cache_entry(key)

    def delete_cache_entry(self, key):
        del self.cache[key]

    def __del__(self):  # Don't trust the gc delete it manually
        self.clear_store()
