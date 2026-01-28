# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import uuid
import datetime
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Glyconnection", layout="centered")

# =====================================================
# SUPABASE CONFIG
# =====================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials not found in environment variables")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================================================
# SESSION TRACKING
# =====================================================
def register_visit():
    try:
        supabase.table("Visits").insert({
            "id": str(uuid.uuid4()),
            "visited_at": datetime.datetime.utcnow().isoformat(),
            "page": "home",
            "user_agent": st.session_state.get("user_agent", "unknown")
        }).execute()
    except Exception as e:
        st.write("Visit log error:")
        st.write(e)

# منع التكرار داخل نفس الـ session
if "visit_logged" not in st.session_state:
    st.session_state.visit_logged = True
    register_visit()
# =====================================================
# UI HEADER
# =====================================================
st.title("Glyconnection")
st.text('Enter your smiles')

# =====================================================
# BASIC UTILITIES
# =====================================================
def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return mol

def calc_descriptors(mol):
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

def lipinski_hits(mol):
    hits = 0
    if Descriptors.MolWt(mol) <= 500:
        hits += 1
    if Descriptors.MolLogP(mol) <= 5:
        hits += 1
    if Descriptors.NumHDonors(mol) <= 5:
        hits += 1
    if Descriptors.NumHAcceptors(mol) <= 10:
        hits += 1
    return hits

# =====================================================
# GLYCOSIDE LOGIC
# =====================================================
def find_anomeric_carbon(sugar_mol):
    for atom in sugar_mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        oxy_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
        if len(oxy_neighbors) >= 2:
            return atom.GetIdx()
    return None

def find_linkage_atoms(aglycone_mol, atom_symbol):
    return [
        atom.GetIdx()
        for atom in aglycone_mol.GetAtoms()
        if atom.GetSymbol() == atom_symbol
    ]

def create_glycoside(aglycone_smiles, sugar_smiles, atom_symbol):
    aglycone = Chem.MolFromSmiles(aglycone_smiles)
    sugar = Chem.MolFromSmiles(sugar_smiles)
    if aglycone is None or sugar is None:
        raise ValueError("Invalid input SMILES")

    sugar = Chem.AddHs(sugar)
    rw_sugar = Chem.RWMol(sugar)

    anomeric_idx = find_anomeric_carbon(sugar)
    if anomeric_idx is None:
        return {"status": "failed", "reason": "No anomeric carbon"}

    anomeric = rw_sugar.GetAtomWithIdx(anomeric_idx)

    oh_idx = None
    for nbr in anomeric.GetNeighbors():
        if nbr.GetSymbol() == "O" and not nbr.IsInRing():
            oh_idx = nbr.GetIdx()
            break

    if oh_idx is None:
        return {"status": "failed", "reason": "No anomeric OH"}

    o_atom = rw_sugar.GetAtomWithIdx(oh_idx)
    o_atom.SetFormalCharge(-1)
    o_atom.SetNumExplicitHs(0)

    sugar = rw_sugar.GetMol()
    sugar = Chem.RemoveHs(sugar)

    combo = Chem.CombineMols(aglycone, sugar)
    rw = Chem.RWMol(combo)

    sugar_offset = aglycone.GetNumAtoms()
    anomeric_new = anomeric_idx + sugar_offset

    linkage_atoms = [
        atom.GetIdx() for atom in aglycone.GetAtoms()
        if atom.GetSymbol() == atom_symbol
    ]

    if not linkage_atoms:
        return {
            "status": "failed",
            "reason": f"No {atom_symbol} atom in aglycone"
        }

    success_smiles = None

    for linkage_idx in linkage_atoms:
        try:
            rw_trial = Chem.RWMol(rw)
            rw_trial.AddBond(linkage_idx, anomeric_new, Chem.BondType.SINGLE)
            rw_trial.RemoveBond(anomeric_new, oh_idx + sugar_offset)
            rw_trial.RemoveAtom(oh_idx + sugar_offset)

            mol = rw_trial.GetMol()
            Chem.SanitizeMol(mol)

            success_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            break

        except Exception:
            continue

    if success_smiles is None:
        return {
            "status": "unstable",
            "smiles": Chem.MolToSmiles(mol, canonical=False),
            "reason": f"{atom_symbol}-linkage not chemically feasible"
        }

    return {
        "status": "ok",
        "smiles": success_smiles
    }

def generate_variants(aglycone, sugar):
    variants = {}
    for atom, name in [("O", "O-glycoside"),
                       ("N", "N-glycoside"),
                       ("S", "S-glycoside"),
                       ("C", "C-glycoside")]:
        try:
            smi = create_glycoside(aglycone, sugar, atom)
            variants[name] = smi
        except:
            continue
    return variants

def score_variant(mol, desc):
    hits = lipinski_hits(mol)
    score = hits * 2 - abs(desc["LogP"] - 2) - 0.01 * desc["TPSA"]
    return round(score, 3)

def explain(best, all_scores):
    explanation = f"""
Best glycosidic linkage: {best}

Reasoning:
"""
    for k, v in all_scores.items():
        explanation += f"""
{k}:
- score = {v['score']}
- LogP = {v['LogP']}
- TPSA = {v['TPSA']}
"""
    return explanation

# =====================================================
# LOAD TRAINING DATA FROM SUPABASE
# =====================================================
@st.cache_data
def load_training_data():
    res = supabase.table("data - delta").select(
        "smiles, glycoside_type, linkage_atom"
    ).execute()
    return pd.DataFrame(res.data)

def analyze_training_data(df):
    stats = {}
    for atom in ["O", "N", "S", "C"]:
        sub = df[df["linkage_atom"] == atom]
        stats[atom] = {
            "count": len(sub),
            "freq": len(sub) / max(len(df), 1)
        }
    return stats

def training_bonus(linkage_atom, training_stats):
    freq = training_stats.get(linkage_atom, {}).get("freq", 0)
    return round(freq * 5, 3)

def score_variant_with_training(mol, desc, linkage_atom, training_stats):
    base = score_variant(mol, desc)
    bonus = training_bonus(linkage_atom, training_stats)
    return round(base + bonus, 3)

def analyze_linkage(aglycone, sugar):
    variants = generate_variants(aglycone, sugar)
    results = {}

    for link_type, smi_info in variants.items():
        if smi_info.get("status") != "ok":
            continue

        try:
            mol = Chem.MolFromSmiles(smi_info["smiles"])
            desc = calc_descriptors(mol)
            score = score_variant(mol, desc)

            results[link_type] = {
                "status": "ok",
                "SMILES": smi_info["smiles"],
                "score": score,
                **desc
            }

        except Exception as e:
            continue

    if not results:
        return None, None

    best = max(results, key=lambda x: results[x]["score"])
    return best, results

# =====================================================
# MAIN PIPELINE
# =====================================================
def run_pipeline(aglycone, sugar):
    best, all_results = analyze_linkage(aglycone, sugar)
    if best is None:
        return None
    explanation = explain(best, all_results)
    supabase.table("results").insert({
        "aglycone": aglycone,
        "sugar": sugar,
        "best_linkage": best,
        "score": float(all_results[best]["score"])
    }).execute()
    return {
        "best": best,
        "all": all_results,
        "explanation": explanation
    }

# =====================================================
# DATABASE UTILITIES
# =====================================================
def exists_in_db(smiles):
    res = supabase.table("molecules").select("*").eq("smiles", smiles).execute()
    return len(res.data) > 0

def save_molecule(smiles, desc):
    supabase.table("molecules").insert({
        "smiles": smiles,
        **desc
    }).execute()

# =====================================================
# STREAMLIT UI
# =====================================================
aglycone = st.text_input("aglycone")
sugar = st.text_input("sugar")

if st.button("Analyze", key="analyze_btn"):
    if not aglycone or not sugar:
        st.error("Please enter both SMILES")
    else:
        # Track user action
        #update_session_activity()
        
        with st.spinner("Analyzing glycoside variants..."):
            try:
                result = run_pipeline(aglycone, sugar)
                if result is None:
                    st.error("Invalid SMILES input")
                else:
                    st.session_state.result = result
            except Exception as e:
                st.error(str(e))

# =====================================================
# RESULTS
# =====================================================
if "result" in st.session_state:
    res = st.session_state.result

    st.success(f"Best linkage: {res['best']}")
    st.subheader("Scientific Explanation")
    st.text(res["explanation"])

    st.subheader("The possible glycosides")
    df = pd.DataFrame(res["all"]).T
    st.dataframe(df)

    st.subheader("Scientific Interpretation")
    st.write(
        f"""
        The {res['best']}-glycosidic linkage shows the best balance between
        molecular weight, polarity (LogP), hydrogen bonding capacity, TPSA, and has the highest score.
        """
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("""<hr>
    <div style="text-align: center; font-size: 14px;">
        © 2026 
        <a href="https://github.com/ahmedshahatb22" target="_blank" style="text-decoration: none;">
            Ahmed Shahat Belal
        </a>
        — GLYCONNECTION
        <a href=" https://doi.org/10.5281/zenodo.18236882" target="_blank" style="text-decoration: none;">
            DOI
        </a>
    </div>
    """,
    unsafe_allow_html=True)




