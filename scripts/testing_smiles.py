from rdkit import Chem

smiles = r"n1c(csc1\[NH]=C(\N)N)c1cccc(c1)N\C(NC)=[NH]\C#N"

mol = Chem.MolFromSmiles(smiles, sanitize=False)
mol.UpdatePropertyCache(strict=False)

for atom in mol.GetAtoms():
    print(
        atom.GetIdx(),
        atom.GetSymbol(),
        atom.GetDegree(),
        atom.GetExplicitValence(),
        atom.GetFormalCharge(),
    )

# Check chemical validity of molecule 
# Also determines properties of molecule needed for RDKit functions
Chem.SanitizeMol(mol)