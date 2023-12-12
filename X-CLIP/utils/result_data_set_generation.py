import os
import json
import torch
from datasets.build import img_norm_cfg
import torchvision.transforms as transforms
from PIL import Image

def save_image(images, folder, file_name, foto_number):
    """
    Saves an image tensor as a JPEG file.
    """
    if images.ndim == 5:
        images = images[0, :, :, :, :]
    image = images[foto_number].cuda()


    # Denormalize
    mean, std = torch.Tensor(img_norm_cfg['mean']).view(3, 1, 1).cuda(), torch.Tensor(img_norm_cfg['std']).view(3, 1, 1).cuda()
    image.mul_(std).add_(mean)  # In-place denormalization
    image = image / 255.0
    image = torch.clamp(image, 0, 1)

    #convert torch image to PIL image
    transform = transforms.ToPILImage(mode="RGB")

    img = transform(image)

    img.save(os.path.join(folder, file_name))

def update_data(b, images, label_id, idx, all_data, text_labels, text_inputs, config, values_5, indices_5):
    """
    Processes and updates data for each batch during validation.
    """
    image_folder = config.DATA.SAVE_OUTPUT_LOCATION #'/cluster/project/infk/cvg/students/buobm/datasets/HoloAssist/results_data_test'
    assert image_folder is not None, "No Output Location defined"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    b, n, t, c, h, w = images.size()
    for i in range(b):
        foto_list = []
        for num_foto in range(t):
            image_file_name = f"image_{idx}_{i}_{num_foto}.jpg"
            save_image(images[i], image_folder, image_file_name, num_foto)
            foto_list.append(image_file_name)
            
        video_data = {
            "image": foto_list,
            "correct_label": text_labels[label_id[i].item()],
            "correct_class_id": label_id[i].item(),
            "predictions": [{"class_id": idx.item() ,"label": text_labels[idx], "probability": val.item()} for idx, val in zip(indices_5[i], values_5[i])]
        }
        
        all_data.append(video_data)

def save_to_json(data, file_name):
    """
    Saves data to a JSON file.
    """
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)
