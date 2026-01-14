# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Glyconnection", layout="centered")
st.title("🧬 Glyconnection")
st.text('Enter your smiles')

# =====================================================
# SUPABASE CONFIG
# =====================================================
SUPABASE_URL = "https://tmprgujzleuiwqszaojg.supabase.co"
SUPABASE_KEY = "sb_publishable_6AaLYxU0X7IXnMWtnOH73A_TWqWjh1R"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================================================
# BASIC UTILITIES
# =====================================================
def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
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
def generate_variants(aglycone, sugar):
    return {
        "O-glycoside": f"{aglycone}O{sugar}",
        "N-glycoside": f"{aglycone}N{sugar}",
        "C-glycoside": f"{aglycone}C{sugar}",
        "S-glycoside": f"{aglycone}S{sugar}"
    }


def score_variant(mol, desc):
    hits = lipinski_hits(mol)
    score = hits * 2 - abs(desc["LogP"] - 2) - 0.01 * desc["TPSA"]
    return round(score, 3)


def analyze_linkage(aglycone, sugar):
    variants = generate_variants(aglycone, sugar)
    results = {}

    for link_type, smi in variants.items():
        try:
            mol = mol_from_smiles(smi)
            desc = calc_descriptors(mol)
            score = score_variant(mol, desc)

            results[link_type] = {
                "SMILES": smi,
                **desc,
                "Lipinski": lipinski_hits(mol),
                "score": score
            }
        except:
            continue

    if not results:
        return None, None

    best = max(results, key=lambda x: results[x]["score"])
    return best, results


def explain(best, all_results):
    explanation = f"""
Best glycosidic linkage: {best}

Scientific reasoning:
"""
    for k, v in all_results.items():
        explanation += f"""
{k}
- score: {v['score']}
- LogP: {v['LogP']}
- TPSA: {v['TPSA']}
"""

    explanation += """
General interpretation:
• O-glycosides → higher solubility
• N-glycosides → metabolic stability
• C-glycosides → chemical stability
• S-glycosides → lipophilicity modulation
"""
    return explanation

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
# STREAMLIT UI
# =====================================================
aglycone = st.text_input("aglycone")
sugar = st.text_input("sugar")

if st.button("Analyze", key="analyze_btn"):
    if not aglycone or not sugar:
        st.error("Please enter both SMILES")
    else:
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

    st.success(f"✴️ Best linkage: {res['best']}")
    st.subheader("Scientific Explanation")
    st.text(res["explanation"])

    st.subheader("All Variants Comparison")
    df = pd.DataFrame(res["all"]).T
    st.dataframe(df)

    


    # =========================
    # Scientific Explanation
    # =========================
    st.subheader("Scientific Interpretation")

    st.write(
        f"""
        The {res['best']}-glycosidic linkage shows the best balance between
        molecular weight, polarity (LogP), hydrogen bonding capacity, and TPSA.

        This suggests improved bioavailability and ADMET profile
        compared to other linkage types.
        """
    )

    # =========================
    # SAVE TO SUPABASE
    # =========================
    supabase.table("results").insert(
        {
            "aglycone": aglycone,
            "sugar": sugar,
            "best_linkage": res["best"],
            "score": float(res["all"][res["best"]]["score"])
        }
    ).execute()

    st.info("Result saved to database")

# ==============================
# Smart Glycoside AI – Core Script
# ==============================

# --------- BASIC UTILS ---------
def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return mol

def calc_descriptors(smiles):
    mol = mol_from_smiles(smiles)
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotB": Descriptors.NumRotatableBonds(mol)
    }

# --------- CHECK DB ---------
def exists_in_db(smiles):
    res = supabase.table("molecules").select("*").eq("smiles", smiles).execute()
    return len(res.data) > 0

def save_molecule(smiles, desc):
    supabase.table("molecules").insert({
        "smiles": smiles,
        **desc
    }).execute()

# --------- GLYCOSIDE GENERATION ---------
def generate_variants(aglycone, sugar):
    variants = {
        "O-glycoside": f"{aglycone}O{sugar}",
        "N-glycoside": f"{aglycone}N{sugar}",
        "S-glycoside": f"{aglycone}S{sugar}",
        "C-glycoside": f"{aglycone}C{sugar}"
    }
    return variants

# --------- SCORING ---------
def score_variant(desc):
    score = (
        -0.01 * desc["MW"] +
        -0.5 * abs(desc["LogP"] - 2) +
        0.3 * desc["HBD"] +
        0.3 * desc["HBA"] +
        -0.01 * desc["TPSA"]
    )
    return round(score, 3)

# --------- LLM EXPLANATION (Rule-based placeholder) ---------
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

    explanation += """
General interpretation:
• O-glycosides → higher solubility
• N-glycosides → metabolic stability
• C-glycosides → highest chemical stability
• S-glycosides → lipophilicity modulation
"""
    return explanation

# --------- MAIN PIPELINE ---------
def run_pipeline(aglycone_smiles, sugar_smiles):

    # Check & store base molecules
    for smi in [aglycone, sugar]:
        if not exists_in_db(smi):
            desc = calc_descriptors(smi)
            save_molecule(smi, desc)

    # Generate variants
    variants = generate_variants(aglycone, sugar)

    results = {}

    for link_type, smi in variants.items():
        try:
            desc = calc_descriptors(smi)
            score = score_variant(desc)

            results[link_type] = {
                "smiles": smi,
                **desc,
                "score": score
            }

            # save each variant
            supabase.table("variants").insert({
                "linkage": link_type,
                "smiles": smi,
                **desc,
                "score": score
            }).execute()

        except:
            continue

    # Choose best
    best = max(results, key=lambda x: results[x]["score"])

    explanation = explain(best, results)

    # Save final result
    supabase.table("results").insert({
        "aglycone": aglycone,
        "sugar": sugar,
        "best_linkage": best,
        "explanation": explanation
    }).execute()

    return {
        "best": best,
        "all": results,
        "explanation": explanation
    }

    
    
st.markdown("""<hr>
    <div style="text-align: center; font-size: 14px;">
        © 2026 
        <a href="https://github.com/ahmedshahatb22" target="_blank" style="text-decoration: none;">
            Ahmed Shahat Belal
        </a>
        — GLYCONNECTION
        <a href=" https://doi.org/10.5281/zenodo.18236882" target="_blank" style="text-decoration: none;">
            DOI
    </div>
    """,

    unsafe_allow_html=True)


def log_visit():
    # أول مرة فقط في الجلسة
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

        supabase.table("visits").insert({
            "session_id": st.session_state.session_id
        }).execute()

log_visit()
