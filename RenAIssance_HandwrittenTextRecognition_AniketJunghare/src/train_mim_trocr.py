import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW
from PIL import Image
import jiwer
from tqdm import tqdm

# =========================
# 1. MIM Head Definition
# =========================
class MIMHead(nn.Module):
    """
    Lightweight decoder head that reconstructs masked image patches
    from encoder embeddings.
    """
    def __init__(self, hidden_size=768, patch_size=16, image_size=384):
        super().__init__()
        # Total number of patches per image
        num_patches = (image_size // patch_size) ** 2
        # MLP decoder: hidden → hidden → (patch_pixels)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, patch_size * patch_size * 3)
        )

    def forward(self, encoder_outputs, mask_indices):
        """
        encoder_outputs: tensor of shape [B, P, H]
        mask_indices: list of 1D LongTensors, one per batch item
        returns: list of reconstructed patches [M, patch_pixels] per item
        """
        reconstructions = []
        for batch_idx, idxs in enumerate(mask_indices):
            hidden = encoder_outputs[batch_idx, idxs]    # [M, H]
            recon = self.decoder(hidden)                 # [M, ps*ps*3]
            reconstructions.append(recon)
        return reconstructions


# ======================================
# 2. Dataset Classes
# ======================================
class HandwritingDataset(Dataset):
    """
    Returns pixel_values tensor and labels tensor for OCR fine-tuning.
    """
    def __init__(self, image_dir, transcription_dir, processor):
        self.image_dir = image_dir
        self.transcription_dir = transcription_dir
        self.processor = processor
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg'))
            and os.path.exists(
                os.path.join(transcription_dir, f.rsplit('.', 1)[0] + '.txt')
            )
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")\
                                 .pixel_values.squeeze(0)

        txt_path = os.path.join(self.transcription_dir, img_name.rsplit('.', 1)[0] + '.txt')
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.processor.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


class MIMDataset(HandwritingDataset):
    """
    Extends HandwritingDataset to output:
      - mask_indices: which patches are masked
      - patch_targets: ground-truth pixel values for masked patches
    """
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        pixels = item["pixel_values"]           # [3, H, W]
        _, H, W = pixels.shape
        ps = 16                                 # patch size
        # 1. Divide image into non-overlapping patches
        patches = pixels.unfold(1, ps, ps).unfold(2, ps, ps)
        patches = patches.permute(1, 2, 0, 3, 4)    # [nH, nW, 3, ps, ps]
        P = (H // ps) * (W // ps)
        patches = patches.reshape(P, 3, ps, ps)     # [P, 3, ps, ps]

        # 2. Randomly choose half of patches to mask
        num_mask = P // 2
        mask_idxs = torch.randperm(P)[:num_mask]

        # 3. Extract ground-truth pixels for masked patches
        targets = patches[mask_idxs].reshape(num_mask, -1)  # [num_mask, ps*ps*3]

        item["mask_indices"]  = mask_idxs
        item["patch_targets"] = targets
        return item


# ============================================
# 3. MIM Pre-Training Function
# ============================================
def run_mim_pretraining(model, processor, image_dir, transcription_dir,
                        device, mim_epochs=20, batch_size=8, lr=3e-5):
    """
    Perform MIM pre-training: only encoder and MIM head learn,
    using pixel MSE loss on masked patches.
    """
    # Move model to device for encoder training
    model.to(device)

    mim_dataset = MIMDataset(image_dir, transcription_dir, processor)
    mim_loader  = DataLoader(mim_dataset, batch_size=batch_size, shuffle=True)

    # Extract integer image size from processor config
    img_size = processor.feature_extractor.size["height"]
    mim_head = MIMHead(
        hidden_size=model.encoder.config.hidden_size,
        patch_size=16,
        image_size=img_size
    ).to(device)

    optimizer = AdamW(
        list(model.encoder.parameters()) + list(mim_head.parameters()), lr=lr
    )
    mse_loss = nn.MSELoss()

    for epoch in tqdm(range(1, mim_epochs + 1), desc="MIM Epochs"):
        model.encoder.train()
        total_loss = 0.0

        for batch in tqdm(mim_loader, desc=f"Epoch {epoch}", leave=False):
            pixels    = batch["pixel_values"].to(device)             # [B,3,H,W]
            mask_idxs = batch["mask_indices"]                        # list of tensors
            targets   = [t.to(device).float() for t in batch["patch_targets"]]

            # 1) Encode masked images
            enc_out = model.encoder(pixel_values=pixels, return_dict=True)
            hidden  = enc_out.last_hidden_state                     # [B, P, H]

            # 2) Reconstruct masked patches
            recons = mim_head(hidden, mask_idxs)

            # 3) Compute average MSE loss
            loss = sum(mse_loss(r, t) for r, t in zip(recons, targets)) / len(targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(mim_loader)
        tqdm.write(f"Epoch {epoch} completed — Avg MSE Loss: {avg:.4f}")

    return model


# ====================================================
# 4. Supervised Fine-Tuning Function
# ====================================================
def run_supervised_finetuning(model, processor, train_dir, train_txt_dir,
                              val_dir, val_txt_dir, device,
                              num_epochs=50, batch_size=16, lr=5e-5):
    """
    Standard TrOCR fine-tuning with tqdm bars for epochs and batches.
    """
    train_ds = HandwritingDataset(train_dir, train_txt_dir, processor)
    val_ds   = HandwritingDataset(val_dir,   val_txt_dir,   processor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    best_cer = float('inf')

    # Set required config parameters for decoding
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id

    for epoch in tqdm(range(1, num_epochs + 1), desc="Supervised Epochs"):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_cer, val_wer = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                pv   = batch["pixel_values"].to(device)
                lbls = batch["labels"].to(device)

                gen   = model.generate(pv, num_beams=5, early_stopping=True)
                preds = processor.batch_decode(gen, skip_special_tokens=True)

                temp = lbls.clone()
                temp[temp == -100] = processor.tokenizer.pad_token_id
                truths = processor.batch_decode(temp, skip_special_tokens=True)

                val_cer += jiwer.cer(truths, preds)
                val_wer += jiwer.wer(truths, preds)

        avg_cer = val_cer / len(val_loader)
        avg_wer = val_wer / len(val_loader)
        print(f"Epoch {epoch}/{num_epochs} — Train Loss: {avg_train:.4f}"
              f" | Val CER: {avg_cer:.4f} | Val WER: {avg_wer:.4f}")

        # Save best model
        if avg_cer < best_cer:
            best_cer = avg_cer
            model.save_pretrained("best_mim_trocr_model")
            processor.save_pretrained("best_mim_trocr_model")
            print(f"Best model saved (CER: {best_cer:.4f})")

    return model



# ======================
# 5. Pipeline
# ======================
if __name__ == "__main__":
    # Device and processor initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten",
        use_fast=True
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    # MIM Pre-Training on unlabeled images
    model = run_mim_pretraining(
        model, processor,
        image_dir="Working_dataset/train",               # Unlabeled images for MIM
        transcription_dir="Working_dataset/train_transcriptions", 
        device=device,
        mim_epochs=3,
        batch_size=16,
        lr=3e-5
    )

    # Standard Supervised Fine-Tuning
    model = run_supervised_finetuning(
        model, processor,
        train_dir="Working_dataset/train",
        train_txt_dir="Working_dataset/train_transcriptions",
        val_dir="Working_dataset/validation",
        val_txt_dir="Working_dataset/validation_transcriptions",
        device=device,
        num_epochs=250,
        batch_size=16,
        lr=5e-5
    )

