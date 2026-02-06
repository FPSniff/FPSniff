
import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler

from DiffusionFreeGuidence.FP_dataset import FingerprintDataset
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as T

import cv2

def calculate_phash_DCT(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.resize(img, (32, 32))
    dct = cv2.dct(np.float32(img))
    dct_low_freq = dct[:8, :8]
    mean_val = np.mean(dct_low_freq[1:, 1:])
    hash_bits = dct_low_freq > mean_val
    hash_str = ''.join(['1' if v else '0' for v in hash_bits.flatten()])
    phash_hex = ''.join(['%x' % int(hash_str[i:i+4], 2) for i in range(0, 64, 4)])
    return phash_hex

def hamming_distance(hash1, hash2):
    if len(hash1) != len(hash2):
        raise ValueError("Hash length mismatch")
    return sum(bin(int(a, 16) ^ int(b, 16)).count('1') for a, b in zip(hash1, hash2))

def hamming_to_similarity(dist, hash_length=64):
    return 1 - (dist / hash_length)

def calculate_ssim(image_path1, image_path2):
    """
    Compute SSIM between two images
    """
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise ValueError(f"Failed to read images for SSIM computation：{image_path1}, {image_path2}")
    # Resize if image sizes do not match
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (min_w, min_h))
        img2 = cv2.resize(img2, (min_w, min_h))
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    img_dir=modelConfig["img_dir"]
    loss_log_path = os.path.join(modelConfig["save_dir"], "train_loss.txt")

    transform = transforms.Compose([

        transforms.Resize((modelConfig["img_size"], modelConfig["img_size"])),  # Resize image
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = FingerprintDataset(img_dir=img_dir, transform=transform)

    
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    
    net_model = UNet(T=modelConfig["T"], num_labels=5, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))
        # Place below the training epoch loop
        with open(loss_log_path, "a") as f:
            f.write(f"{e}\t{loss.item()}\n")
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_dir"], f'ckpt_{e}_.pt'))
        eval_config = {
            **modelConfig,  # Inherit all settings
            "test_img_dir": modelConfig["train_eval_img_dir"],  # You can set eval image dir in modelConfig
            "results_dir": os.path.join(modelConfig["save_dir"], f"eval_epoch_{e}")
        }
        eval_config["test_load_weight"] = f'ckpt_{e}_.pt'
        eval(eval_config)





def add_noise(img, t, sqrt_alphas_bar, sqrt_one_minus_alphas_bar, device):
    noise = torch.randn_like(img)
    a = sqrt_alphas_bar[t].to(device)
    b = sqrt_one_minus_alphas_bar[t].to(device)
    while a.ndim < img.ndim:
        a = a.unsqueeze(-1)
    while b.ndim < img.ndim:
        b = b.unsqueeze(-1)
    return a * img + b * noise


def save_img(img_tensor, path):
    save_image(torch.clamp(img_tensor * 0.5 + 0.5, 0, 1), path)

def eval(modelConfig: dict):
    test_img_dir = modelConfig["test_img_dir"]
    origin_img_dir = modelConfig["origin_img_dir"]
    test_img_paths = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.lower().endswith('.bmp')]
    device = torch.device(modelConfig["device"])

    model = UNet(
        T=modelConfig["T"], num_labels=5, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]
    ).to(device)
    ckpt = torch.load(os.path.join(
        modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    sampler = GaussianDiffusionSampler(
        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]
    ).to(device)

    preprocess = T.Compose([
        T.Resize((modelConfig["img_size"], modelConfig["img_size"])),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

    os.makedirs(modelConfig["results_dir"], exist_ok=True)
    phash_log_path = os.path.join(modelConfig["results_dir"], "phash_similarity.txt")
    guide_lambda = modelConfig.get("guide_lambda", 0.8)
    start_t = modelConfig.get("start_t", modelConfig["T"]-1)

    with open(phash_log_path, "w") as flog:
        flog.write("filename\tphash_enhanced\tphash_origin\thamming_dist\tsimilarity\tssim\n")

    for img_path in test_img_paths:
        original_img_name = os.path.basename(img_path)[:-20]+".bmp"  # Strip suffix
        img = Image.open(img_path).convert("L")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        noise = torch.randn_like(img_tensor)
        labels = torch.ones(1, dtype=torch.long).to(device)
        with torch.no_grad():
            enhanced = sampler(
                noise, labels,
                start_t=start_t,
                guide_img=img_tensor,
                guide_lambda=guide_lambda
            )
            enhanced = enhanced[0].cpu()
            enhanced_save_path = os.path.join(
                modelConfig["results_dir"],
                f"{os.path.splitext(original_img_name)[0]}_guided_sample.png"
            )
            save_img(enhanced, enhanced_save_path)
            print(f"Saved: {enhanced_save_path}")

        # Compute pHash and Hamming distance
        try:
            phash_enhanced = calculate_phash_DCT(enhanced_save_path)
        except Exception as e:
            print(f"[Warning] Enhanced img phash fail: {enhanced_save_path} -- {e}")
            phash_enhanced = "error"
        origin_path = os.path.join(origin_img_dir, original_img_name)
        try:
            phash_origin = calculate_phash_DCT(origin_path)
        except Exception as e:
            print(f"[Warning] Origin img phash fail: {origin_path} -- {e}")
            phash_origin = "error"
        if "error" not in [phash_enhanced, phash_origin]:
            dist = hamming_distance(phash_enhanced, phash_origin)
            sim = hamming_to_similarity(dist, 64)
        else:
            dist, sim = -1, -1

        # Compute SSIM
        try:
            ssim_score = calculate_ssim(enhanced_save_path, origin_path)
        except Exception as e:
            print(f"[Warning] SSIM computation failed: {enhanced_save_path}, {origin_path} -- {e}")
            ssim_score = -1

        # Write to log file
        with open(phash_log_path, "a") as flog:
            flog.write(f"{original_img_name}\t{phash_enhanced}\t{phash_origin}\t{dist}\t{sim:.4f}\t{ssim_score:.4f}\n")

        torch.cuda.empty_cache()

    print(f"Done! Saved all guided enhanced images to {modelConfig['results_dir']}")
    print(f"pHash/SSIM log saved to {phash_log_path}")