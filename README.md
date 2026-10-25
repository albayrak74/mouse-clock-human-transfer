# mouse-clock-human-transfer

Analysis code for:

> Alkan İ, Albayrak M. *Transferring a pan-tissue mouse epigenetic clock to human methylation arrays: laboratory-held-out validation, feature adaptation and few-shot recalibration.* (submitted)

## Data
All data are public.

* **Mouse (training):** RRBS count matrices of the mouse aging series distributed by MethAgingDB (Zenodo, doi: [10.5281/zenodo.15714493](https://doi.org/10.5281/zenodo.15714493)); extract each series into `raw/x/<GSE>/`. GEO series-matrix files for the same series (sample annotation) go in `geo/`.
* **Human (test):** GEO series matrices of GSE20242, GSE41037, GSE15745 (27K) and GSE40279, GSE61258, GSE50498, GSE52401 (450K), plus the GPL13534 platform annotation, in `human/`.
* **Annotation:** NCBI `gene_info` and `gene_orthologs` tables and the mm10 / hg19 GTF files, in `ann/`.

## Pipeline
Run the scripts from the repository root in this order; `ann/`, `geo/`, `raw/`, `human/`, `out/` and `fig/` are relative paths set at the top of each script.

| Step | Script | Output |
|---|---|---|
| 1 | `geo_meta.py` | per-sample GEO annotation of the mouse series |
| 2 | `m1_samples.py` | mouse sample table (age, tissue, strain, condition, use flag) |
| 3 | `m2_ortho.py` | one-to-one human–mouse orthologs and promoter windows (TSS −1500/+500) |
| 4 | `m3_mouse_features.py` | gene-level promoter methylation for every usable mouse sample |
| 5 | `h1_human.py` | gene-level promoter methylation and sample tables for the human arrays |
| 6 | `m6_analysis.py` | within-mouse CV / LODO / LOTO; mouse→human transfer with the four feature adaptations; few-shot recalibration |
| 7 | `m7_followup.py` | clipped back-transform, k-sample-mean baseline, age strata, bootstrap CIs |
| 8 | `m8_sensitivity.py` | sensitivity to the number of genes (k) and the elastic-net penalty |
| 9 | `m9_revision.py` | bound saturation, donor-level CIs, LOGSE and related revision analyses |
| 10 | `v_values.py` | every number quoted in the manuscript → `out/V.json` |
| 11 | `f_figs.py` | Figures 1–3 |

`m4_load.py` (data loaders), `m5_models.py` (clock fitting and feature transforms) and `helpers.py` are imported by the other scripts.

## Software
Python 3.13, scikit-learn 1.9.1, numpy, pandas, scipy, matplotlib. Random seeds are fixed in the scripts.

## License
MIT
