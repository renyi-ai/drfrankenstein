from dotmap import DotMap
from tqdm import tqdm

from src.comparators.hook_store import HookStore


def default_step_callback(model, inputs, targets):
    inputs = inputs.to(model.device)
    model(inputs)


class ComparatorBaseClass:

    def __init__(self, models: DotMap) -> None:
        self.models = models
        [model.eval() for model in self.models.values()]
        self.hook_store = HookStore()

    def iterate_through_data(self, data_loader,
                             group_at=float('inf'),
                             stop_at=float('inf'),
                             step_callback=default_step_callback,
                             accumulate_callback=None):
        processed_sample_num = 0
        n_saves = 0
        for i, (inputs, targets) in tqdm(enumerate(data_loader, start=1),
                                         desc=f"{type(self).__name__} "
                                              f"is iterating through the data",
                                         total=len(data_loader)):
            for model_name, model in self.models.items():
                self.hook_store.active_model_name = model_name
                step_callback(model, inputs, targets)

            # Calculate measures if group is full, then save
            processed_sample_num += inputs.shape[0]
            need_save = processed_sample_num >= group_at
            # check last opportunity
            need_save = need_save or i == len(data_loader)

            if accumulate_callback is not None and need_save:
                processed_sample_num = 0
                n_saves += 1
                accumulate_callback()

            if n_saves == stop_at:
                break
