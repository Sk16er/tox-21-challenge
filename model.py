import torch
import torch.nn as nn
from rdkit import Chem

# =============================================================================
# Featurization Utils
# =============================================================================

# Atom feature sizes
ATOM_FEATURES = {
    'atomic_num': list(range(1, 101)),  # 1-100
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Bond feature sizes
BOND_FDIM = 13  # Based on standard 1-hot

def get_atom_fdim():
    return sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2 # +2 for aromaticity, mass

def get_bond_fdim():
    return BOND_FDIM

def onek_encoding_unk(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def atom_features(atom):
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled mass
    return features

def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0)
    ]
    bond_feats += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return bond_feats

class MolGraph:
    """
    Represents a molecular graph. 
    Constructs the graph structure (f_atoms, f_bonds, connectivity).
    """
    def __init__(self, smiles):
        self.smiles = smiles
        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms = []
        self.f_bonds = []
        self.a2b = [] # mapping from atom index to incoming bond indices
        self.b2a = [] # mapping from bond index to source atom index
        self.b2revb = [] # mapping from bond index to reverse bond index
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            self.n_atoms = mol.GetNumAtoms()
            for atom in mol.GetAtoms():
                self.f_atoms.append(atom_features(atom))
            
            self.a2b = [[] for _ in range(self.n_atoms)]
            
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                
                b_feat = bond_features(bond)
                
                # Bond a1 -> a2
                b1_idx = self.n_bonds
                self.f_bonds.append(b_feat)
                self.b2a.append(a1)
                self.a2b[a2].append(b1_idx) # Incoming for a2
                
                # Bond a2 -> a1
                b2_idx = self.n_bonds + 1
                self.f_bonds.append(b_feat) # Same features
                self.b2a.append(a2)
                self.a2b[a1].append(b2_idx) # Incoming for a1
                
                self.b2revb.append(b2_idx)
                self.b2revb.append(b1_idx)
                
                self.n_bonds += 2

class BatchMolGraph:
    """
    Collates a list of MolGraph objects into a batch for D-MPNN.
    """
    def __init__(self, mol_graphs):
        self.atom_features = []
        self.bond_features = []
        self.a2b = []
        self.b2a = []
        self.b2revb = []
        
        self.a_scope = [] # (start, len) for atoms
        self.b_scope = [] # (start, len) for bonds
        
        total_atoms = 0
        total_bonds = 0
        
        for mol_graph in mol_graphs:
            self.atom_features.extend(mol_graph.f_atoms)
            self.bond_features.extend(mol_graph.f_bonds)
            
            # Shift indices
            for b_list in mol_graph.a2b:
                self.a2b.append([b + total_bonds for b in b_list])
            
            for a in mol_graph.b2a:
                self.b2a.append(a + total_atoms)
            
            for b in mol_graph.b2revb:
                self.b2revb.append(b + total_bonds)
                
            self.a_scope.append((total_atoms, mol_graph.n_atoms))
            self.b_scope.append((total_bonds, mol_graph.n_bonds))
            
            total_atoms += mol_graph.n_atoms
            total_bonds += mol_graph.n_bonds
        
        self.atom_features = torch.tensor(self.atom_features, dtype=torch.float)
        self.bond_features = torch.tensor(self.bond_features, dtype=torch.float)
        self.b2a = torch.tensor(self.b2a, dtype=torch.long)
        self.b2revb = torch.tensor(self.b2revb, dtype=torch.long)
            
    def get_components(self):
        """Returns tensors needed for the forward pass."""
        return (self.atom_features, self.bond_features, self.a2b, self.b2a, self.b2revb, self.a_scope)

# =============================================================================
# D-MPNN Model
# =============================================================================

class DMPNN(nn.Module):
    def __init__(self, hidden_size=300, depth=3, tasks=12, global_feats_size=0):
        super(DMPNN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()
        self.global_feats_size = global_feats_size
        
        # Input Encoders
        self.W_i = nn.Linear(self.atom_fdim + self.bond_fdim, hidden_size, bias=False)
        
        # Message Passing Weights
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(self.atom_fdim + hidden_size, hidden_size)
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1) # Standard dropout
        
        # Readout
        # Concatenate global features (if roughly similar range) to aggregated vector
        self.readout_1 = nn.Linear(hidden_size + global_feats_size, hidden_size)
        self.readout_2 = nn.Linear(hidden_size, tasks)
        
    def forward(self, batch_graph, global_feats=None):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope = batch_graph.get_components()
        
        if f_atoms.shape[0] == 0:
            # Handle empty batch if needed
            return torch.zeros((len(a_scope), self.readout_2.out_features)).to(self.W_i.weight.device)

        device = self.W_i.weight.device
        f_atoms = f_atoms.to(device)
        f_bonds = f_bonds.to(device)
        b2a = b2a.to(device)
        b2revb = b2revb.to(device)

        # 1. Initialize hidden states for edges/bonds
        atom_features_source = f_atoms.index_select(0, b2a)
        input_tensor = torch.cat([atom_features_source, f_bonds], dim=1)
        dataset_h_0 = self.act(self.W_i(input_tensor))
        
        h = dataset_h_0
        
        # 2. Message Passing
        for _ in range(self.depth):
            # Sum all incoming messages to each atom (scattering)
            atom_messages = torch.zeros(f_atoms.shape[0], self.hidden_size).to(device)
            target_atoms = b2a.index_select(0, b2revb)
            atom_messages.index_add_(0, target_atoms, h)
            
            # Select messages for each bond's source atom
            atom_messages_source = atom_messages.index_select(0, b2a)
            
            # Subtract reverse message
            m = atom_messages_source - h.index_select(0, b2revb)
            
            # Update h
            h = self.act(dataset_h_0 + self.W_h(m))
            h = self.dropout(h)
        
        # 3. Readout
        target_atoms = b2a.index_select(0, b2revb)
        atom_messages = torch.zeros(f_atoms.shape[0], self.hidden_size).to(device)
        atom_messages.index_add_(0, target_atoms, h)
        
        final_atom_input = torch.cat([f_atoms, atom_messages], dim=1)
        atom_h = self.act(self.W_o(final_atom_input))
        
        mol_vecs = []
        for (start, size) in a_scope:
            if size == 0:
                mol_vecs.append(torch.zeros(self.hidden_size).to(device))
            else:
                mol_h = atom_h.narrow(0, start, size)
                mol_vecs.append(mol_h.sum(dim=0))
        
        mol_vecs = torch.stack(mol_vecs, dim=0)
        
        # Concatenate global features (RDKit descriptors)
        if self.global_feats_size > 0:
            if global_feats is None:
                raise ValueError("Model initialized with global features but none provided in forward()")
            global_feats = global_feats.to(device)
            mol_vecs = torch.cat([mol_vecs, global_feats], dim=1)
        
        output = self.readout_1(mol_vecs)
        output = self.act(output)
        output = self.dropout(output)
        logits = self.readout_2(output)
        
        return logits
