import os
import warnings
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW
from tqdm import tqdm
import jiwer
import torch.cuda.amp
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore", message="Using a slow image processor")

class HandwritingDataset(Dataset):
    def __init__(self, image_dir, transcription_dir, processor, max_length=512):
        self.image_dir = image_dir
        self.transcription_dir = transcription_dir
        self.processor = processor
        self.max_length = max_length
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png')) and os.path.exists(
                os.path.join(transcription_dir, f.rsplit('.', 1)[0] + '.txt')
            )
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        txt_path = os.path.join(self.transcription_dir, img_name.rsplit('.', 1)[0] + '.txt')

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()

        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Replace padding token id with -100 for loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

# Initialize with fast processor
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten",
    use_fast=True
)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Datasets and loaders
train_dataset = HandwritingDataset(
    image_dir="Working_dataset/train",
    transcription_dir="Working_dataset/train_transcriptions",
    processor=processor
)
val_dataset = HandwritingDataset(
    image_dir="Working_dataset/validation",
    transcription_dir="Working_dataset/validation_transcriptions",
    processor=processor
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training setup
num_epochs = 250
best_cer = float('inf')
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # Training loop with mixed precision
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_cer, val_wer = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Generate predictions
            outputs = model.generate(
                pixel_values,
                max_length=model.config.max_length,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode predictions
            pred_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Prepare labels for decoding
            labels = labels.clone()  # Create a copy to avoid modifying original
            labels[labels == -100] = processor.tokenizer.pad_token_id
            labels = labels.clamp(0, len(processor.tokenizer) - 1)
            
            try:
                true_texts = processor.batch_decode(labels, skip_special_tokens=True)
            except Exception as e:
                print(f"Error decoding batch: {e}")
                print(f"Label stats - min: {labels.min()}, max: {labels.max()}")
                continue
            
            # Calculate metrics
            val_cer += jiwer.cer(true_texts, pred_texts)
            val_wer += jiwer.wer(true_texts, pred_texts)
    
    val_cer /= len(val_loader)
    val_wer /= len(val_loader)
    scheduler.step(val_cer)
    
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f}")
    
    # Save best model
    if val_cer < best_cer:
        best_cer = val_cer
        model.save_pretrained("output/best_trocr_model")
        processor.save_pretrained("output/best_trocr_model")
        print(f"New best model saved (CER: {val_cer:.4f})")
    
    # Periodic checkpoints
    if (epoch + 1) % 250 == 0:
        model.save_pretrained(f"output/trocr-checkpoint-epoch{epoch+1}")
