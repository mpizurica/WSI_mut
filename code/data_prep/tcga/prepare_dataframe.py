import pandas as pd
import os

if __name__ == '__main__':
    gene = 'TP53' 
    only_grade = 'all' # only_grade = 'highest'
    GS_sort = ['GSTumor', 'GS3+3', 'GS3+4', 'GS4+3', 'GS4+4', 'GS3+5', 'GS4+5', 'GS5+3','GS5+4', 'GS5+5']
    df_name = 'df_patients_labels.csv'

    ############# mutation labels
    tcga_df = pd.read_csv('TCGA.PRAD.mutect.deca36be-bf05-441a-b2e4-394228f23fbe.DR-10.0.somatic.maf',
                            comment='#',
                            sep='\t',
                            low_memory=False,
                            skip_blank_lines=True,
                            header=0)

    # tcga mutect
    tcga_df = tcga_df[tcga_df['Hugo_Symbol']==gene]
    tcga_df = tcga_df[tcga_df['Variant_Type']=='SNP']
    tcga_df = tcga_df[['Tumor_Sample_Barcode','Hugo_Symbol','Variant_Type', 'IMPACT']]
    tcga_df['patient_id'] = tcga_df['Tumor_Sample_Barcode'].str[:12]
    tcga_patients_gene = tcga_df['patient_id'].values

    # our labels
    df_snvs = pd.read_csv('PRAD_pyclone_snvs.tsv', sep='\t')
    df_indels = pd.read_csv('PRAD_indels_annotated.tsv', sep='\t')
    df_snvs_gene = df_snvs[df_snvs['gene']==gene]
    df_indels_gene = df_indels[df_indels['gene']==gene]
    our_patients_gene=list(df_snvs[df_snvs['gene']==gene]['sample_id'].values)+list(df_indels[df_indels['gene']==gene]['sample_id'].values)

    # find inconsistent labels
    inconsistent = []

    for p in our_patients_gene:
        if p not in tcga_patients_gene:
            inconsistent.append(p)

    for p in tcga_patients_gene:
        if p not in our_patients_gene:
            inconsistent.append(p)

    ############# bad qual slides
    bad_qual = pd.read_csv('bad_qual_slides.txt', header=None)
    bad_qual_slides = bad_qual[0].values
    
    # slides will contain slides of good quality
    slides = []

    sources = ['/data/thesis_2021/PRAD_dx/tiles_512px'] # folder with tiles (subtree as in README)
    for source in sources:
        
        for patient_subfolder in os.listdir(source):
            if patient_subfolder not in bad_qual_slides:
                slides.append(patient_subfolder)

    # remove slides of patients with inconsitent labels        
    slides_new = [i for i in slides if i[:12] not in inconsistent]

    ############# gather data in dataframe (all tiles, patient ids and their label)
    patients_labels = []
    tile_name = []

    for source in sources:
        
        # source folders contain folders with name = name of slide of patient
        for patient_subfolder in os.listdir(source):
            
            if patient_subfolder in slides_new:
                
                if patient_subfolder[:12] in tcga_patients_gene:
                    label = 1
                else:
                    label = 0
            
                # patient_subfolder contains subfolders per gleason score, tiles are in these folders
                for GS_subfolder in os.listdir(source+'/'+patient_subfolder):
                    if GS_subfolder != 'Image': #sometimes qupath makes an additional export 'Image'
                        for tile_path in os.listdir(source+'/'+patient_subfolder+'/'+GS_subfolder):

                            patients_labels.append([patient_subfolder[:12], source+'/'+patient_subfolder+'/'+GS_subfolder+'/'+tile_path, label])
                            tile_name.append('TCGA'+tile_path.split('TCGA')[-1])
                            
            #else:
                #print('no label for '+patient_subfolder[:12])

    df_patients_labels = pd.DataFrame(patients_labels)
    df_patients_labels.columns = ['patient_id', 'tile_path', gene]
    df_patients_labels['tile_name'] = tile_name

    df_patients_labels.to_csv(df_name)


    



    