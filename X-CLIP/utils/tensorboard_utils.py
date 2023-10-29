# Helper functions for tensorboard visualization
import torch
from datasets.build import img_norm_cfg
from yacs.config import CfgNode as CN

class ClassificationMetricsLogger:
    def __init__(self, num_classes):
        self.TP = torch.zeros(num_classes).cuda()
        self.FP = torch.zeros(num_classes).cuda()
        self.FN = torch.zeros(num_classes).cuda()

    def update(self, pred_indices, true_labels):
        # Count the occurrences of each label
        for cls_idx in range(self.TP.shape[0]):
            self.TP[cls_idx] += torch.sum((pred_indices == cls_idx) & (true_labels == cls_idx))
            self.FP[cls_idx] += torch.sum((pred_indices == cls_idx) & (true_labels != cls_idx))
            self.FN[cls_idx] += torch.sum((pred_indices != cls_idx) & (true_labels == cls_idx))

    def write_to_tensorboard(self, writer, epoch):
        precision = self.TP / (self.TP + self.FP + 1e-10)
        recall = self.TP / (self.TP + self.FN + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        avg_precision = torch.mean(precision)
        avg_recall = torch.mean(recall)
        avg_f1_score = torch.mean(f1_score)

        writer.add_scalar('Validation/Precision', avg_precision.item(), epoch)
        writer.add_scalar('Validation/Recall', avg_recall.item(), epoch)
        writer.add_scalar('Validation/F1_Score', avg_f1_score.item(), epoch)

    def reset(self):
        self.TP.zero_()
        self.FP.zero_()
        self.FN.zero_()

def add_imagages_to_tensorboard(writer, image, image_number, epoch, name):
    # only take one view
    if image.ndim == 6:
        image = image[:, 0, :, :, :, :]
    frame_img = image[image_number].squeeze(0)[0].cuda()  # Averaging [C, H, W]

    # Denormalize
    mean, std = torch.Tensor(img_norm_cfg['mean']).view(3, 1, 1).cuda(), torch.Tensor(img_norm_cfg['std']).view(3, 1, 1).cuda()
    frame_img.mul_(std).add_(mean)  # In-place denormalization

    # Clip values to [0, 1]
    frame_img =  frame_img / 255.0
    frame_img = torch.clamp(frame_img, 0, 1)
    # Move tensor to CPU and add to TensorBoard
    denorm_img = frame_img.cpu()
    writer.add_image(name, denorm_img, epoch)


def get_hparams(config, max_accuracy):
    # Extract hyperparameters
    hparams = {
        'ARCH': config.MODEL.ARCH,
        'DROP_PATH_RATE': config.MODEL.DROP_PATH_RATE,
        'FIX_TEXT': config.MODEL.FIX_TEXT,
        'PRETRAINED': config.MODEL.PRETRAINED,
        'RESUME': config.MODEL.RESUME,
        'NUM_FRAMES': config.DATA.NUM_FRAMES,
        'NUM_CLASSES': config.DATA.NUM_CLASSES,
        'NUM_CROPS': config.TEST.NUM_CROP,
        'NUM_CLIPS': config.TEST.NUM_CLIP,
        'NUM_VIEWS': config.TEST.NUM_CROP * config.TEST.NUM_CLIP,
        'DATASET': config.DATA.DATASET,
        'ONLY_TEST': config.TEST.ONLY_TEST,
        'EPOCHS': config.TRAIN.EPOCHS,
        'WARMUP_EPOCHS': config.TRAIN.WARMUP_EPOCHS,
        'WEIGHT_DECAY': config.TRAIN.WEIGHT_DECAY,
        'LR': config.TRAIN.LR,
        'BATCH_SIZE': config.TRAIN.BATCH_SIZE,
        'LR_SCHEDULER': config.TRAIN.LR_SCHEDULER,
        'OPTIMIZER': config.TRAIN.OPTIMIZER,
        'OPT_LEVEL': config.TRAIN.OPT_LEVEL,
        'LABEL_SMOOTH': config.AUG.LABEL_SMOOTH,
        'COLOR_JITTER': config.AUG.COLOR_JITTER,
        'GRAY_SCALE': config.AUG.GRAY_SCALE,
        'MIXUP': config.AUG.MIXUP,
        'CUTMIX': config.AUG.CUTMIX,
        'MIXUP_SWITCH_PROB': config.AUG.MIXUP_SWITCH_PROB
    }
    # Metrics
    metrics = {
        'hparam/accuracy': max_accuracy
    }
    return hparams, metrics