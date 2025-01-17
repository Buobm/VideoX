import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler, update_learning_rate
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, generate_class_list
from datasets.build import build_dataloader, img_norm_cfg
from utils.logger import create_logger
from utils.tensorboard_utils import ClassificationMetricsLogger, get_hparams, add_imagages_to_tensorboard, log_system_resources, perform_tsne, ConfusionMatrixLogger
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip, clip_mean
from utils.result_data_set_generation import update_data, save_to_json




def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default= int(os.environ['LOCAL_RANK']), help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config): 
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    if config.MODEL.NAME == 'XCLIP':
        model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                            device="cpu", jit=False, 
                            T=config.DATA.NUM_FRAMES, 
                            droppath=config.MODEL.DROP_PATH_RATE, 
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                            use_cache=config.MODEL.FIX_TEXT,
                            logger=logger,
                            use_text_prompts=config.MODEL.USE_TEXT_PROMPTS,
                            num_classes= config.DATA.NUM_CLASSES,
                            )
    elif config.MODEL.NAME == 'CLIPMEAN':
        assert config.TEST.ONLY_TEST, "CLIP model can only be used for inference"
        model = clip_mean.CLIPBenchmark(config.MODEL.ARCH) 
    else:
        raise ValueError(f"Unknown model: {config.MODEL.NAME}")

    model = model.cuda()

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES, 
                                       smoothing=config.AUG.LABEL_SMOOTH, 
                                       mixup_alpha=config.AUG.MIXUP, 
                                       cutmix_alpha=config.AUG.CUTMIX, 
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)


    text_labels = generate_text(train_data)
    
    # perform_tsne(model, text_labels, train_data.classes, writer, tag = f"Labels {config.MODEL.ARCH}")

    if config.TEST.ONLY_TEST:
        acc1, acc5 = validate(val_loader, text_labels, model, config, train_data=train_data, epoch=0, confusion_matrix_log=True)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        parameters, metric = get_hparams(config,acc1, acc5)
        writer.add_hparams(parameters, metric)
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        acc1 = 0.0
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, acc1)

        acc1, acc5 = validate(val_loader, text_labels, model, config, train_data=train_data, epoch=epoch)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    # config.defrost()
    # config.TEST.NUM_CLIP = 4
    # config.TEST.NUM_CROP = 3
    # config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    acc1, acc5 = validate(val_loader, text_labels, model, config, train_data=train_data, epoch= config.TRAIN.EPOCHS + 1, confusion_matrix_log=True)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
    parameters, metric = get_hparams(config,acc1, acc5)
    writer.add_hparams(parameters, metric)

def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, val_acc):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    texts = text_labels.cuda(non_blocking=True)
    
    for idx, batch_data in enumerate(train_loader):

        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        images = images.view((-1,config.DATA.NUM_FRAMES,3)+images.size()[-2:])
        
        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        
        output = model(images, texts)

        total_loss = criterion(output, label_id)
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0 or idx == num_steps - 1:
                optimizer.step()
                optimizer.zero_grad()
                update_learning_rate(lr_scheduler, epoch, num_steps, idx, val_acc)
                #lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            update_learning_rate(lr_scheduler, epoch, num_steps, idx, val_acc)
            #lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            writer.add_scalar('Train/Learning Rate', lr, epoch * num_steps + idx)
            writer.add_scalar('Train/Total Loss', tot_loss_meter.avg, epoch * num_steps + idx)
            log_system_resources(writer, epoch * num_steps + idx)
        
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(val_loader, text_labels, model, config, train_data, epoch=0, confusion_matrix_log=True):
    model.eval()
    all_data = []
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    # Initialize the classification logger
    metrics_logger = ClassificationMetricsLogger(num_classes=config.DATA.NUM_CLASSES)
    if confusion_matrix_log:
        confusion_matrix_logger = ConfusionMatrixLogger(train_data=train_data)
    class_names = generate_class_list(train_data)
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)
           
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                
                output = model(image_input, text_inputs)
                
                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)

            if config.DATA.SAVE_OUTPUT_LOCATION is not None:
                update_data(b, _image, label_id, idx, all_data, class_names, text_inputs, config, values_5, indices_5)

            acc1, acc5 = 0, 0
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                elif (config.TEST.ONLY_TEST or epoch == config.TRAIN.EPOCHS + 1) and random.random() < 0.1:
                    # Add 10% of the wrongly classified images to TensorBoard
                    # only in last epoch or when testing
                    add_imagages_to_tensorboard(writer, _image, i, epoch, f'Wrongly_Classified/{label_id[i]}')
                if label_id[i] in indices_5[i]:
                    acc5 += 1
            # Update TP, FP, FN 
            metrics_logger.update(indices_1, label_id)
            if confusion_matrix_log:
                confusion_matrix_logger.update(indices_1, label_id)
            
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    writer.add_scalar('Validation/Accuracy@1', acc1_meter.avg, epoch)
    writer.add_scalar('Validation/Accuracy@5', acc5_meter.avg, epoch)
    if confusion_matrix_log:
        confusion_matrix_logger.generate_confusion_matrix(writer, epoch)
    metrics_logger.write_to_tensorboard(writer, epoch)
    metrics_logger.reset()
    
    if config.DATA.SAVE_OUTPUT_LOCATION is not None:
        save_to_json(all_data, f'{config.DATA.SAVE_OUTPUT_LOCATION}/output_data.json')

    return acc1_meter.avg, acc5_meter.avg

if __name__ == '__main__':
    print("Starting X-CLIP")
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    global writer
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.MODEL.ARCH + "__" + str(config.DATA.NUM_FRAMES)
    run_name = run_name.replace("/", "_")
    writer = SummaryWriter(log_dir=config.OUTPUT + '/tensorboard/' + run_name + "_" + current_date)
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)
    writer.close()