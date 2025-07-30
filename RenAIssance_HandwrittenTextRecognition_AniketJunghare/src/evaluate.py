import os
import time
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from tqdm import tqdm
from jiwer import wer, cer

# ==== Config ====
model_dir = "best_trocr_model"
test_image_dir = "Working_dataset/test"
test_transcription_dir = "Working_dataset/test_transcriptions"
output_csv_path = "output/trocr_predictions_final.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== Load model ====
processor = TrOCRProcessor.from_pretrained(model_dir)
model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
model.eval()

# ==== Gather test images ====
image_files = sorted([
    f for f in os.listdir(test_image_dir)
    if f.lower().endswith(('.jpg', '.png')) and os.path.exists(
        os.path.join(test_transcription_dir, os.path.splitext(f)[0] + ".txt")
    )
])

# ==== Storage for DataFrame ====
rows = []

# ==== Inference ====
print("\n Running inference and building dataframe...\n")

for image_file in tqdm(image_files, desc="Processing"):
    image_path = os.path.join(test_image_dir, image_file)
    transcription_path = os.path.join(test_transcription_dir, os.path.splitext(image_file)[0] + ".txt")

    # Load image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Run beam search
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            num_beams=5,
            num_return_sequences=5,
            early_stopping=True,
            max_length=processor.tokenizer.model_max_length,
            output_scores=True,
            return_dict_in_generate=True
        )

    # Decode predictions
    pred_texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    pred_texts = [t.strip() for t in pred_texts]

    # Compute logprobs
    sequence_logprobs = []
    for i in range(len(outputs.sequences)):
        logprob = 0.0
        for t, score_dist in enumerate(outputs.scores):
            if t < len(outputs.sequences[i]):
                token_id = outputs.sequences[i][t]
                logprob += torch.log_softmax(score_dist[i], dim=-1)[token_id].item()
        sequence_logprobs.append(round(logprob, 2))

    # Load GT
    with open(transcription_path, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()

    # Append row to DataFrame
    rows.append({
        "image_name": image_file,
        "ground_truth": gt_text,
        "predictions": pred_texts,
        "logprobs": sequence_logprobs
    })

# ==== Create and Save DataFrame ====
df = pd.DataFrame(rows)
df.to_csv(output_csv_path, index=False)
print(f"\n Saved CSV to: {output_csv_path}")
