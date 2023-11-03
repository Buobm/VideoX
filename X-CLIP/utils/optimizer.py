import copy
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.distributed as dist

def is_main_process():
    return dist.get_rank() == 0

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}]


def fix_text(model):
    for name, param in model.named_parameters():
        if "visual." in name or "mit" in name or "prompts" in name:
            continue
        else:
            param.requires_grad=False

def build_optimizer(config, model):
    model = model.module if hasattr(model, 'module') else model
    
    # fix text
    if config.MODEL.FIX_TEXT:
        fix_text(model)
    
    # set decay and lr
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    clip_parameters = set_weight_decay(model, skip, skip_keywords, 
        weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR, 
        have=(), not_have=("prompts", "mit", "message_")
    )
    msg_parameters = set_weight_decay(model, skip, skip_keywords,
        weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10, 
        have=("message_",), not_have=()
    )
    mit_parameters = set_weight_decay(model, skip, skip_keywords,
        weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10, 
        have=("mit",), not_have=()
    )
    prompts_parameters = set_weight_decay(model, skip, skip_keywords, 
        weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10, 
        have=("prompts",), not_have=()
    )

    # Choose optimizer based on config
    if config.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(clip_parameters + mit_parameters + prompts_parameters + msg_parameters,
                        betas=(0.9, 0.98), eps=1e-8,)
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(clip_parameters + mit_parameters + prompts_parameters + msg_parameters,
                        betas=(0.9, 0.98), eps=1e-8,)
    else:
        raise ValueError(f"Unsupported optimizer: {config.TRAIN.OPTIMIZER}")
   
    return optimizer


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    
    if config.TRAIN.LR_SCHEDULER == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.LR / 100,
            warmup_lr_init=0,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER == 'stepdecay':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.TRAIN.STEP_SIZE,
            gamma=config.TRAIN.LR_DECAY,
        )
    elif config.TRAIN.LR_SCHEDULER == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max', 
            factor=config.TRAIN.LR_DECAY,
            patience=config.TRAIN.PATIENCE,
            verbose=True,
            threshold=1e-4,
            min_lr=config.TRAIN.LR / 100
        )
    else:
        raise ValueError(f"Unsupported learning rate scheduler: {config.TRAIN.LR_SCHEDULER}")

    return lr_scheduler

def update_learning_rate(lr_scheduler, epoch, num_steps, index ,val_loss=None):
    """
    Update the learning rate based on the type of the provided scheduler.

    Args:
    - lr_scheduler: The learning rate scheduler object.
    - val_loss: The validation loss. Required if using 'ReduceLROnPlateau'.
    - current_step: Current overall step (usually epoch * num_steps + idx). Required if using a custom scheduler like 'CosineLRScheduler'.
    - index: Required to check for new epoch if using STEP LR scheduler or ReduceLROnPlateau scheduler.

    Returns:
    - None
    """
    if index == num_steps -1:
        print("index == num_steps -1")
    if isinstance(lr_scheduler, optim.lr_scheduler.StepLR):
        if index == num_steps -1:
            lr_scheduler.step()
            print("step")
    elif isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if val_loss is None:
            raise ValueError("val_loss is required for ReduceLROnPlateau scheduler")
        if index == num_steps -1:
            lr_scheduler.step(val_loss)
    else:
        # Assuming all other custom schedulers like CosineLRScheduler have step_update method
        lr_scheduler.step_update(epoch * num_steps + index)
