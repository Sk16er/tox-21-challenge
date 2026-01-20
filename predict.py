import torch
import os
import numpy as np
import logging
import glob
import pickle
from model import DMPNN, MolGraph, BatchMolGraph
from data import TASKS, get_global_features, SCALER_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Tox21Predictor:
    def __init__(self, model_dir="."):
        self.models = []
        self.scaler = None
        
        # Load scaler
        if os.path.exists(SCALER_FILE):
             with open(SCALER_FILE, 'rb') as f:
                self.scaler = pickle.load(f)
             logger.info(f"Loaded scaler from {SCALER_FILE}")
        else:
            logger.warning("Scaler not found. Global features will may be unscaled (or 0).")
            
        # Find all model seeds in checkpoints
        model_paths = glob.glob(os.path.join("checkpoints", "model_seed_*.pt"))
        if not model_paths:
             # Fallback to local
             model_paths = glob.glob("model_seed_*.pt")
             
        if not model_paths:
             # Fallback to legacy single model
             if os.path.exists("best_model.pt"):
                 model_paths = ["best_model.pt"]
             else:
                 logger.warning("No models found!")
        
        for path in model_paths:
            # We need to instantiate model with correct dimensions.
            # Assuming hidden=300, depth=3.
            # Global feats size depends on scaler
            global_feats_size = self.scaler.n_features_in_ if self.scaler else 0
            
            model = DMPNN(hidden_size=300, depth=3, tasks=len(TASKS), global_feats_size=global_feats_size)
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(path))
            else:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            
            model.to(DEVICE)
            model.eval()
            self.models.append(model)
            logger.info(f"Loaded model from {path}")
            
    def predict(self, smiles_list):
        """
        Predicts toxicity for a list of SMILES using ensemble consensus.
        """
        import os # Re-importing locally to ensure safety if copied
        
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        valid_smiles = []
        valid_indices = []
        mol_graphs = []
        global_features_list = []
        
        from rdkit import Chem # Ensure rdkit available
        
        for i, s in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(s)
            mg = MolGraph(s)
            if mg.n_atoms > 0 and mol is not None:
                mol_graphs.append(mg)
                # Compute global features
                feats = get_global_features(mol)
                global_features_list.append(feats)
                
                valid_smiles.append(s)
                valid_indices.append(i)
        
        results = {}
        # Handle invalid inputs
        for i, s in enumerate(smiles_list):
            if i not in valid_indices:
                results[s] = None

        if not mol_graphs:
            return results

        # Normalize features
        if global_features_list:
            global_features_arr = np.stack(global_features_list)
            if self.scaler:
                global_features_arr = self.scaler.transform(global_features_arr)
            global_features_tensor = torch.tensor(global_features_arr, dtype=torch.float32)
        else:
            global_features_tensor = torch.zeros((0, 0)) # Should not happen if mol_graphs not empty

        batch_graph = BatchMolGraph(mol_graphs)
        
        # Ensemble Inference
        ensemble_preds = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(batch_graph, global_features_tensor)
                preds = torch.sigmoid(logits).cpu().numpy()
                ensemble_preds.append(preds)
        
        # Average predictions
        if ensemble_preds:
            avg_preds = np.mean(ensemble_preds, axis=0)
        else:
            return results # No models loaded

        # Map back results
        for idx_in_batch, original_idx in enumerate(valid_indices):
            s = smiles_list[original_idx]
            pred_dict = {
                task: float(avg_preds[idx_in_batch, i]) 
                for i, task in enumerate(TASKS)
            }
            results[s] = pred_dict
            
        return results

if __name__ == "__main__":
    predictor = Tox21Predictor()
    test_smiles = ["CCO", "c1ccccc1", "InvalidSmiles"]
    print(predictor.predict(test_smiles))
