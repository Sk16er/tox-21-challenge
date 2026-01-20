# to do - add coments[/]
from data import load_data, Tox21Dataset
from train import collate_fn  
from predict import Tox21Predictor
from sklearn.metrics import roc_auc_score
import numpy as np

def test_strength():
    print("Testing model strength (Ensemble Performance)...")
    # 1. Loading the validation dataset
    _, val_data, _ = load_data() 
    
    # 2. Initialize the predictor 
    predictor = Tox21Predictor()
    
    all_preds = []
    all_labels = []
    
    # 3. every molecule prediction
    for item in val_data:
        smiles = item['smiles']
        labels = item['labels'].numpy()
        
        # Predict
        res = predictor.predict([smiles])
        if res[smiles] is not None:
            pred_vec = np.array([res[smiles][t] for t in res[smiles]])
            all_preds.append(pred_vec)
            all_labels.append(labels)
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 4. Calculate AUC per task
    task_aucs = []
    for i in range(12):
        valid_idx = all_labels[:, i] != -1
        if sum(all_labels[valid_idx, i]) > 0: #bug fix
            auc = roc_auc_score(all_labels[valid_idx, i], all_preds[valid_idx, i])
            task_aucs.append(auc)
    
    avg_auc = np.mean(task_aucs)
    print(f"\n--- LOCAL TEST RESULTS ---")
    print(f"Average ROC-AUC: {avg_auc:.4f}")
    print(f"Strength Rating: {'TOP 5 READY' if avg_auc > 0.83 else 'great but not in top5'}")
     

if __name__ == "__main__":
    test_strength()