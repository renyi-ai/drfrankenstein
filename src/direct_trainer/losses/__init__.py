def get_loss(loss_str):
    if loss_str == "mse":
        from torch.nn import MSELoss
        loss = MSELoss(reduction="none")
    if loss_str == "cka":
        from src.comparators.compare_functions.cka_torch import cka
        loss = cka
    return loss
