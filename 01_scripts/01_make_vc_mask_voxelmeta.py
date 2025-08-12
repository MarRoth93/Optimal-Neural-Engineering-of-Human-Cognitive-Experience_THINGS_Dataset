import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting
from pathlib import Path

# === Pfade ===
BRAINMASK = Path("/home/rothermm/THINGS/02_data/derivatives/fmriprep/sub-01/anat/sub-01_space-T1w_mask.nii.gz")
VOXEL_TSV = Path("/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_voxel-metadata.tsv")
OUT_MASK  = Path("/home/rothermm/THINGS/02_data/masks/subj01/sub-01_visualcortex_mask.nii.gz")
OUT_PNG   = Path("/home/rothermm/THINGS/02_data/masks/subj01/sub-01_visualcortex_mask_overlay.png")

# === 1) Brainmask laden ===
bm_img = nib.load(str(BRAINMASK))
bm = bm_img.get_fdata().astype(bool)
shape, aff, hdr = bm.shape, bm_img.affine, bm_img.header

# === 2) Voxel-Metadaten laden ===
df = pd.read_csv(VOXEL_TSV, sep=",")

# === 3) Visuelle Spalten definieren (aus Header) ===
vc_cols = [
    "V1","V2","V3","hV4","VO1","VO2",
    "LO1 (prf)","LO2 (prf)","TO1","TO2","V3b","V3a",
    "lEBA","rEBA","lFFA","rFFA","lOFA","rOFA",
    "lSTS","rSTS","lPPA","rPPA","lRSC","rRSC",
    "lTOS","rTOS","lLOC","rLOC","IT"
]

# Falls Spaltennamen Sonderzeichen enthalten (z.B. "LO1 (prf)"), genau so übernehmen
missing = [c for c in vc_cols if c not in df.columns]
if missing:
    raise ValueError(f"Fehlende ROI-Spalten im TSV: {missing}")

# === 4) Selektionsmaske auf Basis dieser Spalten ===
sel_mask = (df[vc_cols].sum(axis=1) > 0).values
print(f"[INFO] {sel_mask.sum()} / {len(df)} Voxel als visueller Kortex markiert.")

# === 5) Leeres Volumen anlegen ===
vc_vol = np.zeros(shape, dtype=np.uint8)

# === 6) Voxelkoordinaten setzen ===
I = df.loc[sel_mask, "voxel_x"].astype(int).values
J = df.loc[sel_mask, "voxel_y"].astype(int).values
K = df.loc[sel_mask, "voxel_z"].astype(int).values

# Bounds check
valid = (I>=0)&(I<shape[0])&(J>=0)&(J<shape[1])&(K>=0)&(K<shape[2])
I, J, K = I[valid], J[valid], K[valid]
vc_vol[I, J, K] = 1

# Optional: nur innerhalb Brainmask behalten
vc_vol = vc_vol * bm.astype(np.uint8)

# === 7) Speichern ===
vc_img = nib.Nifti1Image(vc_vol, aff, hdr)
nib.save(vc_img, str(OUT_MASK))
print(f"[OK] Maske gespeichert unter {OUT_MASK.resolve()}")

# === 8) Visualisierung ===
display = plotting.plot_roi(vc_img, bg_img=bm_img, title="sub-01 Visual Cortex Mask (T1w space)")
display.savefig(str(OUT_PNG))
display.close()
print(f"[OK] Overlay gespeichert unter {OUT_PNG.resolve()}")
