import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import logging
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = "tox21.csv"
SCALER_FILE = "scaler.pkl"

TASKS = [
    'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

class Tox21Dataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of dict): List of molecule data (smiles, features, labels, global_features).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_global_features(mol):
    """
    Computes RDKit descriptors for a molecule.
    Returns a numpy array of float32.
    """
    if mol is None:
        return np.zeros(len(Descriptors.descList), dtype=np.float32)
    
    features = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            # Handle infinity or nan
            if not np.isfinite(val):
                val = 0.0
            
            # Clamp to float32 range to avoid overflow
            if val > 3.4e38:
                val = 3.4e38
            elif val < -3.4e38:
                val = -3.4e38
                
            features.append(val)
        except:
            features.append(0.0)
    
    return np.array(features, dtype=np.float32)

def generate_scaffold(mol, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a molecule."""
    mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
    if mol is None:
        return "invalid"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def scaffold_split(df, seed=42, frac_train=0.9, frac_val=0.1):
    """
    Splits the data based on scaffolds.
    Strictly Train/Val split.
    """
    np.random.seed(seed)
    scaffolds = defaultdict(list)
    
    logger.info("Generating scaffolds...")
    for idx, row in df.iterrows():
        smiles = row['smiles']
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    # Sort scaffolds by size (largest first) to balance the split
    scaffold_sets = [scaffolds[s] for s in sorted(scaffolds, key=lambda x: (len(scaffolds[x]), x), reverse=True)]

    train_idxs, val_idxs = [], []
    train_cutoff = frac_train * len(df)

    for scaffold_set in scaffold_sets:
        if len(train_idxs) + len(scaffold_set) <= train_cutoff:
            train_idxs.extend(scaffold_set)
        else:
            val_idxs.extend(scaffold_set)

    logger.info(f"Split sizes: Train={len(train_idxs)}, Val={len(val_idxs)}")
    
    # Validation check
    assert len(set(train_idxs).intersection(set(val_idxs))) == 0, "Overlap detected between train and val!"
    
    return train_idxs, val_idxs

def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Generative download is disabled. Please provide {path} in the working directory.")
    
    df = pd.read_csv(path)
    
    # Pre-filter and compute features
    valid_idxs = []
    mols = []
    all_global_features = []
    
    logger.info("Computing global descriptors...")
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            valid_idxs.append(idx)
            mols.append(mol)
            all_global_features.append(get_global_features(mol))
    
    all_global_features = np.stack(all_global_features) # (N, n_features)
    
    logger.info(f"Original size: {len(df)}, Valid Size: {len(valid_idxs)}")
    df = df.iloc[valid_idxs].reset_index(drop=True)

    # Ensure task columns are numeric
    for task in TASKS:
        df[task] = pd.to_numeric(df[task], errors='coerce')

    # Scaffold Split (90/10)
    train_idx, val_idx = scaffold_split(df, frac_train=0.9, frac_val=0.1)
    
    # Scale Features based on TRAIN set only
    logger.info("Scaling global descriptors...")
    train_features = all_global_features[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    # Save scaler for inference
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    
    all_global_features_scaled = scaler.transform(all_global_features)

    # helper to format data
    def format_subset(indices):
        data_list = []
        for i, original_idx in enumerate(indices):
            # indices are into the *re-indexed* df (where we dropped invalid)
            # which aligns with all_global_features (which also skipped invalid)
            
            # Wait, scaffold_split returns indices into the df passed to it.
            # df was reset_index'd. So 'indices' align with df rows 0..N
            # And all_global_features aligns with df rows 0..N
            
            row = df.iloc[original_idx]
            labels = row[TASKS].values.astype(np.float32)
            labels = np.nan_to_num(labels, nan=-1.0)
            
            global_feats = all_global_features_scaled[original_idx]

            data_list.append({
                'smiles': row['smiles'],
                'labels': torch.tensor(labels, dtype=torch.float32),
                'global_features': torch.tensor(global_feats, dtype=torch.float32),
                'mol_id': row.get('mol_id', f'id_{original_idx}')
            })
        return data_list

    train_data = format_subset(train_idx)
    val_data = format_subset(val_idx)
    
    return train_data, val_data, scaler

if __name__ == "__main__":
    t, v, s = load_data()
    print(f"Loaded {len(t)} train, {len(v)} val samples.")
    print(f"Global feature size: {len(t[0]['global_features'])}")
