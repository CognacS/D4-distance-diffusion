from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from src.data.filters import reg_filters

RDLogger.DisableLog('rdApp.*')

@reg_filters.register()
class FilterInvalidPosMol:
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def __call__(self, mol):
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=self.random_seed, useRandomCoords=True)
            AllChem.UFFOptimizeMolecule(mol)
            return True
        except Exception as e:
            return False


@reg_filters.register()
class FilterQuickInvalidPosMol:
    
    def __init__(self, max_embed_attempts=1, max_opt_iters=100, random_seed=42):
        self.max_embed_attempts = max_embed_attempts
        self.max_opt_iters = max_opt_iters
        self.random_seed = random_seed
    
    def __call__(self, mol):

        # optional: quickly skip tiny molecules or purely ionic/metal-ligand species
        # e.g. if you want to skip molecules with >1 fragment or 1-atom fragments:
        # frags = Chem.GetMolFrags(mol, asMols=True)
        # if any(f.GetNumAtoms() <= 1 for f in frags):
        #     return False

        # add Hs and attempt a single quick embed with limited attempts
        mol_h = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(
            mol_h,
            randomSeed=self.random_seed,
            useRandomCoords=True,
            maxAttempts=self.max_embed_attempts
        )
        if res != 0:           # non-zero -> failed to embed
            return False

        # do a short UFF optimization to detect badly strained (non-converging) cases
        opt_res = AllChem.UFFOptimizeMolecule(mol_h, maxIters=self.max_opt_iters)
        # opt_res == 0 usually means converged; non-zero indicates failure/too strained
        return opt_res == 0