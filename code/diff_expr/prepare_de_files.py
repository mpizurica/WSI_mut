import pandas as pd
import matplotlib.pyplot as plt
import _pickle as cPickle

def TumorVSNormal(sample_info):
    tumor_ids = list(sample_info["sample_submitter_id"][sample_info["sample_type"]=='Primary Tumor'])
    normal_ids = list(sample_info["sample_submitter_id"][sample_info["sample_type"]=='Solid Tissue Normal'])
    # If multiple tumors from the same patients, only keep one
    
    prim = pd.DataFrame(tumor_ids)
    prim.columns = ['sample_submitter_id']
    prim["sample_name"] = prim['sample_submitter_id'].str[:-4]
    prim["sample_num"] = prim['sample_submitter_id'].str[-4:]
    prim = prim.sort_values(by=["sample_name", "sample_num"])
    tumor_ids = list(prim.groupby('sample_name').first()['sample_submitter_id'])
    return tumor_ids, normal_ids

if __name__ == '__main__':
    # TCGA (downloaded from Xenahubs)
    tcga = pd.read_csv("TCGA-PRAD.htseq_counts.tsv.gz", sep="\t")
    tcga = tcga.set_index("Ensembl_ID")
    tcga = tcga.drop_duplicates(keep="first")
    tcga = tcga.loc[[x for x in tcga.index if "ENSG" in x]]

    clin_prad = pd.read_csv("clinical.project-TCGA-PRAD.2022-05-12.tar.gz", sep="\t")
    sample_prad = pd.read_csv("biospecimen.project-TCGA-PRAD.2022-05-12.tar.gz", sep='\t')

    tumor_ids, normal_ids = TumorVSNormal(sample_prad)
    tcga = tcga[[x for x in tcga.columns if x in tumor_ids]]

    groups = pd.read_csv("/project_antwerp/TCGA_training/diff_expr/fp_tn/tn_fp.csv", index_col=0)

    tcga_counts = 2**tcga-1
    col = [x[:-4] for x in tcga_counts.columns]
    tcga_counts.columns = col

    group1 = set(groups["tn"])
    group2 = set(groups["fp"])
    group1_counts = tcga_counts[[x for x in tcga_counts.columns if x in list(group1)]]
    group2_counts = tcga_counts[[x for x in tcga_counts.columns if x in list(group2)]]

    group1_counts.to_csv("tn.tsv", sep="\t", header=True, index=True)
    group2_counts.to_csv("fp.tsv", sep="\t", header=True, index=True)





