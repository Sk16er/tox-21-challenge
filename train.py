import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import logging
from tqdm import tqdm

from data import load_data, Tox21Dataset, TASKS
from model import DMPNN, MolGraph, BatchMolGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
HIDDEN_SIZE = 300
DEPTH = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 43, 44, 45, 46] # Ensemble seeds

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tox21 D-MPNN Ensemble")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs per model")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    """
    Custom collate function to handle graph batching and global features.
    """
    smiles_list = [item['smiles'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    global_features = torch.stack([item['global_features'] for item in batch])
    mol_ids = [item['mol_id'] for item in batch]
    
    # Create MolGraphs
    mol_graphs = [MolGraph(s) for s in smiles_list]
    batch_graph = BatchMolGraph(mol_graphs)
    
    return batch_graph, labels, global_features, mol_ids

def compute_loss(logits, targets, pos_weights=None):
    mask = (targets != -1).float()
    safe_targets = targets.clone()
    safe_targets[safe_targets == -1] = 0
    
    if pos_weights is not None:
        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
    loss = criterion(logits, safe_targets)
    loss = loss * mask
    
    safe_denominator = mask.sum()
    if safe_denominator == 0:
        return torch.tensor(0.0).to(logits.device)
    
    return loss.sum() / safe_denominator

def calculate_pos_weights(train_data):
    labels = []
    for item in train_data:
        labels.append(item['labels'].numpy())
    labels = np.stack(labels)
    
    weights = []
    for i in range(len(TASKS)):
        task_labels = labels[:, i]
        valid = task_labels != -1
        valid_labels = task_labels[valid]
        n_pos = (valid_labels == 1).sum()
        n_neg = (valid_labels == 0).sum()
        
        if n_pos > 0:
            weight = n_neg / n_pos
        else:
            weight = 1.0
        weights.append(weight)
        
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_graph, labels, global_feats, _ in loader:
            logits = model(batch_graph, global_feats)
            preds = torch.sigmoid(logits)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    task_aucs = []
    for i, task in enumerate(TASKS):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        
        mask = y_true != -1
        y_true_valid = y_true[mask]
        y_pred_valid = y_pred[mask]
        
        if len(np.unique(y_true_valid)) == 2:
            try:
                auc = roc_auc_score(y_true_valid, y_pred_valid)
                task_aucs.append(auc)
            except ValueError:
                pass
        else:
            pass
            
    if len(task_aucs) == 0:
        return 0.0
    
    return np.mean(task_aucs)

def train_model(train_loader, val_loader, pos_weights, args, seed, global_feats_size):
    """Trains a single model instance."""
    set_seed(seed)
    model = DMPNN(hidden_size=HIDDEN_SIZE, depth=DEPTH, tasks=len(TASKS), global_feats_size=global_feats_size)
    model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    
    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_seed_{seed}.pt")
    
    best_val_auc = 0.0
    
    logger.info(f"Training Model Seed {seed}...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_graph, labels, global_feats, _ in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1}", leave=False):
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(batch_graph, global_feats)
            loss = compute_loss(logits, labels, pos_weights=pos_weights)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        val_auc = evaluate(model, val_loader)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_path)
            
    logger.info(f"Seed {seed} Best Val AUC: {best_val_auc:.4f}")
    return best_val_auc

def train_ensemble():
    args = parse_args()
    
    logger.info("Loading data (Training/Validation)...")
    train_data, val_data, scaler = load_data()
    
    global_feats_size = len(train_data[0]['global_features'])
    logger.info(f"Global feature dim: {global_feats_size}")
    
    logger.info("Calculating task weights...")
    pos_weights = calculate_pos_weights(train_data)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    best_aucs = []
    
    for seed in SEEDS:
        best_auc = train_model(train_loader, val_loader, pos_weights, args, seed, global_feats_size)
        best_aucs.append(best_auc)
        
    avg_ensemble_auc = np.mean(best_aucs)
    logger.info("="*30)
    logger.info(f"Ensemble Training Complete.")
    logger.info(f"Seed AUCs: {best_aucs}")
    logger.info(f"Average Ensemble Val AUC: {avg_ensemble_auc:.4f}")
    logger.info("="*30)

if __name__ == "__main__":
    train_ensemble()
