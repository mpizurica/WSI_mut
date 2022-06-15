import pandas as pd

if __name__ == '__main__':
    DEgenes = pd.read_csv("DE_fp_tn.tsv", sep="\t")

    significant = DEgenes[(DEgenes["logFC"].abs()>=1)&(DEgenes["adj.P.Val"]<0.05)]
    significant = significant.reset_index()
    significant = significant.rename(columns={'index':'gene_id'})

    # convert ens ids
    conversion_df = pd.read_csv("gencode.gene.info.v22.tsv", sep="\t")
    significant = significant.merge(conversion_df[['gene_id', 'gene_name']].drop_duplicates(), on='gene_id')
    significant.to_csv('DE_fp_vs_tn.csv')