import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset", "kegg")

TEXT_FILE = os.path.join(DATASET_DIR, "drug_text.csv")

ENTITY2ID_FILE = os.path.join(DATASET_DIR, "entity2id.txt")

LOCAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "PubMedBERT")

OUTPUT_FILE = os.path.join(DATASET_DIR, "drug_features.npy")

def get_total_node_count(entity2id_path):

    max_id = 0
    if not os.path.exists(entity2id_path):
        print(f"Warning: {entity2id_path} not found. Using default size 400000.")
        return 400000

    with open(entity2id_path, 'r') as f:

        first_line = f.readline().strip()
        parts = first_line.split('\t')
        if len(parts) == 1 and parts[0].isdigit():
            return int(parts[0])

        try:
            if len(parts) >= 2:
                max_id = max(max_id, int(parts[1]))
        except:
            pass

        for line in f:
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    max_id = max(max_id, int(parts[1]))
            except:
                continue
    return max_id + 1


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_nodes = get_total_node_count(ENTITY2ID_FILE)
    print(f"Total nodes in full graph: {num_nodes}")

    if not os.path.exists(TEXT_FILE):
        print(f"Error: {TEXT_FILE} not found. Please run extraction script first.")
        return

    df = pd.read_csv(TEXT_FILE)

    df['Full_Text'] = df['Full_Text'].fillna("")
    id2text = dict(zip(df['Dataset_ID'], df['Full_Text']))
    valid_ids = set(df['Dataset_ID'].tolist())

    print(f"Loading local model: {LOCAL_MODEL_PATH} ...")
    if not os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        print("Error: Local model files missing!")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model.to(device)
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    feature_dim = 768

    print("Initializing full graph feature matrix (all zeros)...")
    try:
        features_matrix = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    except MemoryError:
        print("Memory Error: Unable to allocate full matrix.")
        return

    model.eval()
    batch_size = 64
    drug_id_list = list(valid_ids)

    print(f"Starting feature generation for {len(drug_id_list)} drugs...")

    for i in tqdm(range(0, len(drug_id_list), batch_size)):
        batch_ids = drug_id_list[i: i + batch_size]
        batch_texts = []
        batch_indices = []

        for idx in batch_ids:
            text = id2text.get(idx, "")
            if len(text) > 5:
                batch_texts.append(text)
                batch_indices.append(idx)

        if not batch_texts:
            continue

        try:

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            features_matrix[batch_indices] = embs

        except Exception as e:
            print(f"Batch error: {e}")
            if "CUDA out of memory" in str(e):
                print("CUDA OOM: Please reduce batch_size.")
                return
            continue

    np.save(OUTPUT_FILE, features_matrix)
    print(f"\nSuccess! Features saved to: {OUTPUT_FILE}")
    print(f"Matrix shape: {features_matrix.shape}")

if __name__ == "__main__":
    main()