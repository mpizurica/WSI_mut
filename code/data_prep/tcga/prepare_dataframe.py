import pandas as pd
import os

if __name__ == '__main__':
    gene = 'TP53' 
    only_grade = 'all' # only_grade = 'highest' #--> choose whether to immediately only keep tiles from region with highest GS
    df_name = 'df_patients_labels.csv'
    source = '/data/thesis_2021/PRAD_dx/tiles_512px' # folder containing slides and tiles (see readme for folder structure)

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

    for slide_subfolder in os.listdir(source):
        if slide_subfolder not in bad_qual_slides:
            slides.append(slide_subfolder)

    # remove slides of patients with inconsitent labels        
    slides_new = [i for i in slides if i[:12] not in inconsistent]

    ############# gather data in dataframe (all tiles, patient ids and their label)
    patients_labels = []
    tile_name = []
        
    # source folders contain folders with name = name of slide of patient
    for slide_subfolder in os.listdir(source):
        
        if slide_subfolder in slides_new:
            
            if slide_subfolder[:12] in tcga_patients_gene:
                label = 1
            else:
                label = 0
        
            # slide_subfolder contains subfolders per gleason score, tiles are in these folders
            for GS_subfolder in os.listdir(source+'/'+slide_subfolder):
                if GS_subfolder != 'Image': #sometimes qupath makes an additional export 'Image'
                    for tile_path in os.listdir(source+'/'+slide_subfolder+'/'+GS_subfolder):

                        patients_labels.append([slide_subfolder[:12], source+'/'+slide_subfolder+'/'+GS_subfolder+'/'+tile_path, label])
                        tile_name.append('TCGA'+tile_path.split('TCGA')[-1])
                        
        #else:
            #print('no label for '+slide_subfolder[:12])

    df_patients_labels = pd.DataFrame(patients_labels)
    df_patients_labels.columns = ['patient_id', 'tile_path', gene]
    df_patients_labels['tile_name'] = tile_name

    ############# only keep tiles from region with highest grade (if desired)
    df_patients_labels['tile_name'] = df_patients_labels['tile_path'].str.split('/').str[-1]
    df_patients_labels['GS'] = df_patients_labels['tile_name'].str.split('_').str[-2]

    GS_map = {'GS3+3':1,'GS3+4':2,'GS4+3':3,'GS4+4':4,\
            'GS3+5':5,'GS4+5':6,'GS5+3':7,'GS5+4':8,\
            'GS5+5':9, 'GS5+3MUCINOUSCARCINOMA':7, 'Tumor':0}

    df_patients_labels['grade'] = df_patients_labels.GS.map(GS_map)

    if only_grade == 'highest':
        dataframes = []

        grouped = df_patients_labels.groupby(df_patients_labels['patient_id'])
        for k,g in grouped:
            g = g.reset_index()
            dataframes.append(g[g['grade'] == g['grade'].max()])

        df_patients_labels_highest = pd.concat(dataframes).reset_index(drop=True) 
        df_patients_labels_highest.to_csv(df_name)
        
    else:
        df_patients_labels.to_csv(df_name)


    



    