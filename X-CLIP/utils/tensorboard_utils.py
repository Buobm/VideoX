# Helper functions for tensorboard visualization
import torch
import torchvision
import psutil
from datasets.build import img_norm_cfg
from yacs.config import CfgNode as CN
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import io
import PIL

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

def log_system_resources(writer, global_step):
    # Log RAM usage (percentage)
    ram_percent = psutil.virtual_memory().percent
    writer.add_scalar('System/RAM_Usage_Percent', ram_percent, global_step)

    # Log RAM usage (absolute in GB)
    ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
    writer.add_scalar('System/RAM_Usage_GB', ram_used_gb, global_step)

    # Log VRAM usage (GB)
    vram_usage_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    writer.add_scalar('System/VRAM_Usage_GB', vram_usage_gb, global_step)

def perform_tsne(model, text_tokens, class_data, writer):
    """
    Perform t-SNE on the embeddings generated by the XCLIP model and log them to TensorBoard.
    
    Parameters:
    - xclip_model: The XCLIP model instance.
    - text_labels: Tokenized text labels to be encoded using the XCLIP model.
    - log_dir: Directory where TensorBoard logs will be saved.
    
    Returns:
    None. The function will write directly to TensorBoard logs.
    """

    classes = [c for i, c in class_data]
    text_tokens = text_tokens.cuda()

    # Handle DDP wrapped models
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
        
    embeddings = model.encode_text(text_tokens)

    writer.add_embedding(embeddings, metadata=classes, tag='Text Embeddings')

class ConfusionMatrixLogger:
    """
    Generate confusion matrices from predictions and true labels and log them to TensorBoard.
    """
    def __init__(self, train_data):
        class_data = train_data.classes 
        self.classes = [c for i, c in class_data]
        self.num_classes = len(self.classes)
        self.true_labels = []
        self.pred_labels = []

    def update(self, pred_indices, true_labels):
        """
        Update the confusion matrix with the provided predictions and true labels.
        
        Parameters:
        - pred_indices: A tensor of shape (N, ) containing the predicted class indices.
        - true_labels: A tensor of shape (N, ) containing the true class indices.
        
        Returns:
        None. The function will update the confusion matrix.
        """
        self.pred_labels.extend(pred_indices.cpu().tolist())
        self.true_labels.extend(true_labels.cpu().tolist())

    def generate_confusion_matrix(self, writer, epoch):
        """
        Generate the confusion matrix and add it to TensorBoard.
        """

        self.log_top_confusions(20,writer, epoch)
        cm = confusion_matrix(self.true_labels, self.pred_labels)
        figure = self.plot_confusion_matrix(cm, class_names=self.classes)
        cm_image = self.plot_to_image(figure)
        writer.add_image("Validation/Confusion Matrix", cm_image, epoch)
        self.reset()
    
    def plot_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        
        buf = io.BytesIO()
        
        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format='png')
        
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        
        image = PIL.Image.open(buf)
        image = torchvision.transforms.ToTensor()(image )#.unsqueeze(0)
        
        return image
    
    def plot_confusion_matrix(self, cm, class_names, threshold=0.05):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        
        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        threshold (float): minimum normalized value to display the confusion
        """
        # Normalize the confusion matrix.
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            normalized_cm = np.nan_to_num(normalized_cm) 
        
        # Create a mask to only show confusions above a certain threshold.
        mask = normalized_cm >= threshold

        figure = plt.figure(figsize=(20, 20))
        plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
        plt.grid(False)  # Hide the grid lines for better clarity

        for i, j in zip(*np.where(mask)):
            plt.text(j, i, f'{normalized_cm[i, j]:.2f}',
                    horizontalalignment="center",
                    color="white" if normalized_cm[i, j] > (cm.max() / 2.) else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure
    
    def top_confusions(self, num_top=10):
        """
        Find the top X confusions from the confusion matrix.

        Args:
        top_x (int): Number of top confusions to return.

        Returns:
        A list of tuples indicating the most confused class pairs.
        Each tuple contains (true_label, predicted_label, count).
        """
        cm = confusion_matrix(self.true_labels, self.pred_labels)
        
        # We're interested only in the non-diagonal elements, so we zero the diagonal.
        np.fill_diagonal(cm, 0)
        
        # Flatten the matrix to 1D array for argsort
        flat_cm = cm.flatten()
        
        # Get the indices of the top X values
        indices = np.argpartition(flat_cm, -num_top)[-num_top:]
        
        # Since argpartition doesn't guarantee sorted order, we sort the indices
        indices = indices[np.argsort(flat_cm[indices])][::-1]
        
        # Map flat indices back to 2D indices
        rows, cols = np.unravel_index(indices, cm.shape)

        # Retrieve class names and counts for the top confusions
        confusions = [(self.classes[row], self.classes[col], cm[row, col]) for row, col in zip(rows, cols)]
        
        return confusions
    
    def log_top_confusions(self, top_x, writer, epoch):
        """
        Log the top X confusions to TensorBoard as text.

        Args:
        top_x (int): Number of top confusions to log.
        writer (SummaryWriter): The TensorBoard summary writer.
        global_step (int): Global step value to tag the summary with.
        """
        confusions = self.top_confusions(top_x)
        
        confusions = self.top_confusions(top_x)
        confusion_str = "Top {} Confusions:\n```\n".format(top_x)
        for true, predicted, count in confusions:
            confusion_str += "True: {:20} | Predicted: {:20} | Count: {}\n".format(true, predicted, count)
        confusion_str += "```"
        
        writer.add_text("Validation/Top Confusions", confusion_str, epoch)


    def reset(self):
        self.true_labels = []
        self.pred_labels = []
        

def get_hparams(config, acc1, acc5):
    # Extract hyperparameters
    hparams = {
        'ARCH': config.MODEL.ARCH,
        'DROP_PATH_RATE': config.MODEL.DROP_PATH_RATE,
        'FIX_TEXT': config.MODEL.FIX_TEXT,
        'PRETRAINED': config.MODEL.PRETRAINED,
        'RESUME': config.MODEL.RESUME,
        'USE_TEXT_PROMPTS': config.MODEL.USE_TEXT_PROMPTS,
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
        'STEP_SIZE': config.TRAIN.STEP_SIZE,
        'LR_DECAY': config.TRAIN.LR_DECAY,
        'PATIENCE': config.TRAIN.PATIENCE,
        'LABEL_SMOOTH': config.AUG.LABEL_SMOOTH,
        'COLOR_JITTER': config.AUG.COLOR_JITTER,
        'GRAY_SCALE': config.AUG.GRAY_SCALE,
        'MIXUP': config.AUG.MIXUP,
        'CUTMIX': config.AUG.CUTMIX,
        'MIXUP_SWITCH_PROB': config.AUG.MIXUP_SWITCH_PROB
    }
    # Metrics
    metrics = {
        'hparam/accuracy@1': acc1,
        'hparam/accuracy@5': acc5
    }
    return hparams, metrics