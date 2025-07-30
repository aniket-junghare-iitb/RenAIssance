import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import ast

# ==== Configuration ====
csv_path = "/home/aniketj/soc/test/trocr_predictions_final.csv"
image_folder = "/home/aniketj/soc/CODE/test"
clip_model_name = "openai/clip-vit-base-patch32"
alpha = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Data ====
df = pd.read_csv(csv_path)
df["predictions"] = df["predictions"].apply(ast.literal_eval)
df["logprobs"] = df["logprobs"].apply(ast.literal_eval)

# ==== Load CLIP ====
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# ==== Process Each Image ====
for idx, row in df.iterrows():
    image_path = f"{image_folder}/{row['image_name']}"
    pred_texts = row["predictions"]
    logprobs = row["logprobs"]

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {image_path} — {e}")
        continue

    # CLIP Inference
    inputs = clip_processor(text=pred_texts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        clip_probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().tolist()

    # Normalize logprobs
    logprobs_np = np.array(logprobs)
    norm_logprobs = (logprobs_np - logprobs_np.min()) / (logprobs_np.max() - logprobs_np.min() + 1e-8)
    final_scores = alpha * np.array(clip_probs) + (1 - alpha) * norm_logprobs

    # Rerank predictions
    reranked = sorted(
        zip(pred_texts, logprobs, clip_probs, final_scores),
        key=lambda x: x[3], reverse=True
    )

    # ==== Output ====
    print("-------------------------------------------------------------------------------------------")
    print(f"\n Image: {row['image_name']}")
    print(f" Ground Truth         : {row['ground_truth']}\n")

    # Top prediction
    top_pred, top_lp, top_cp, top_score = reranked[0]
    print(f" Top Prediction       : {top_pred}")
    print(f"    LogProb           : {top_lp:.2f}")
    print(f"    CLIP Score        : {top_cp:.4f}")
    print(f"    Combined Score    : {top_score:.4f}")

    # Show all predictions
    print("\n All Predictions:")
    for i, (pred, lp, cp, score) in enumerate(reranked):
        print(f"{i+1}. {pred}")
        print(f"    LogProb: {lp:.2f},  CLIP: {cp:.4f},  Score: {score:.4f}")
    print("-------------------------------------------------------------------------------------------")
