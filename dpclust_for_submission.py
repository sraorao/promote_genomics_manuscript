"""module for Dirichlet Process clustering and plotting

Returns:
    [type]: [description]
"""

import pandas
from sklearn.mixture import BayesianGaussianMixture
import glob
import re
import matplotlib.pyplot as plt
import numpy
import seaborn
import itertools
import os
import math
numpy.random.seed(123)

def dpclust(data: 'pandas.DataFrame', weight_concentration_prior: float=None, num_clusters: int=20, inits: int=1000, seed: int=123):
    """Wrapper around sklearn.mixture.BayesianGaussianMixture(); prepares input data, passes it through BayesianGaussianMixture() 
    and returns the clusters called as a 1D numpy array.

    Args:
        data ('pandas.DataFrame'): a dataframe with only the CCF values (samples as column names, each row is an SNV)
        weight_concentration_prior (float, optional): aka gamma. Defaults to None.
        num_clusters (int, optional): Number of clusters to call. Defaults to 20.
        inits (int, optional): Number of initializations to perform. Defaults to 1000.
        seed (int, optional): Seed for random number generator. Defaults to 123.clusters

    Returns:
        [type]: [description]
    """
    print("Clustering data...")
    data_array = numpy.array(data)
    model = BayesianGaussianMixture(n_components=num_clusters, n_init=inits, weight_concentration_prior=weight_concentration_prior, max_iter=int(1e6), random_state=seed, verbose=1, verbose_interval=1000)
    cluster_names = model.fit_predict(data_array)
    print(f'Finished clustering with {inits} iterations')
    return cluster_names

def load_data(dpinfo_files: list, dropna=True):
    """Load data from allDirichletProcessInfo.txt files, adjust for copy number and return all data as a single dataframe.

    Args:
        dpinfo_files (list): List of file paths, perhaps as output from glob.glob()

    Returns:
        pandas DataFrame: A list of hex colour codes.
    """
    print("Loading data from files...")
    tumour_fracs = pandas.concat([pandas.read_csv(x, sep='\t')['mutation.copy.number'] for x in dpinfo_files], axis=1)
    cnadjustment = pandas.concat([pandas.read_csv(x, sep='\t')['no.chrs.bearing.mut'] for x in dpinfo_files], axis=1)
    sample_names = [re.sub('.*/', '', x).replace('_allDirichletProcessInfo.txt', '').replace('-', '_') for x in dpinfo_files]
    tumour_fracs.columns = sample_names
    cnadjustment.columns = sample_names
    cnadjustment = cnadjustment.astype(float)
    tumour_fracs_adjusted = tumour_fracs/cnadjustment
    if dropna:
        tumour_fracs_adjusted = tumour_fracs_adjusted.replace([-numpy.inf, numpy.inf], numpy.nan).dropna()
    return tumour_fracs_adjusted

def load_data_with_locs(dpinfo_files: list, dropna=True):
    """Load data from allDirichletProcessInfo.txt files, adjust for copy number and return all data as a single dataframe.

    Args:
        dpinfo_files (list): List of file paths, perhaps as output from glob.glob()

    Returns:
        pandas DataFrame: A list of hex colour codes.
    """
    print("Loading data from files...")
    tumour_fracs = pandas.concat([pandas.read_csv(x, sep='\t', index_col=['chr', 'start', 'end'])['mutation.copy.number'] for x in dpinfo_files], axis=1)
    cnadjustment = pandas.concat([pandas.read_csv(x, sep='\t', index_col=['chr', 'start', 'end'])['no.chrs.bearing.mut'] for x in dpinfo_files], axis=1)
    sample_names = [re.sub('.*/', '', x).replace('_allDirichletProcessInfo.txt', '').replace('-', '_') for x in dpinfo_files]
    tumour_fracs.columns = sample_names
    cnadjustment.columns = sample_names
    cnadjustment = cnadjustment.astype(float)
    tumour_fracs_adjusted = tumour_fracs/cnadjustment
    if dropna:
        tumour_fracs_adjusted = tumour_fracs_adjusted.replace([-numpy.inf, numpy.inf], numpy.nan).dropna()
    return tumour_fracs_adjusted

def custom_palette(n: int=20):
    """Create a palette of n distinct colours

    Args:
        n (int, optional): Number of colours to return. Defaults to 20.

    Returns:
        list: A list of hex colour codes.
    """
    if n <= 20:
        # https://sashamaps.net/docs/resources/20-colors/
        custom_pal = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', 
                    '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', 
                    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
        # seaborn.palplot(custom_pal[:n])
        return custom_pal[:n]
    else:
        return seaborn.color_palette("Spectral", n_colors=n)[:]
        
def create_pairs(df: 'pandas.DataFrame'):
    """Create pairs from the columns of a dataframe

    Args:
        df (pandas.DataFrame): Columns should contain only sample names

    Returns:
        list of tuples: A list of pairs.
    """
    print("Creating pairs of samples...")
    combos = list(itertools.combinations(df.columns, 2))
    return combos

def plot_clusters(df: 'pandas.DataFrame', exclude_cols: list=['cluster_name'], plot_file: str="", setlimits=True, limit: float=2.5, origin: float=0.0):
    """Draw paired cluster plots for the given dataset.

    Args:
        df (pandas.DataFrame): Samples in columns, cluster names in a column named 'cluster_name'.
        exclude_cols (list, optional): Columns to exclude. Defaults to ['annotation'].

    Returns:
        Matplotlib fig object: Matplotlib fig object
    """
    print("Plotting SNVs...")
    # plt.ioff()
    exclude_cols = [x for x in exclude_cols if x in df.columns]
    combos = create_pairs(df.drop(columns=exclude_cols))
    num_plots = len(combos)
    num_clusters = len(df['cluster_name'].unique())

    fig, ax = plt.subplots(figsize=(20, 4*num_plots))
    # ax.set_box_aspect(1)
    plot_index = 1
    for each in combos:
        plt.subplot(math.ceil(num_plots/2), 2, plot_index, aspect='equal')
        sp = seaborn.scatterplot(data=df, x=each[0], y=each[1], hue=df['cluster_name'].astype(str), 
                                 palette=custom_palette(num_clusters))
        if setlimits:
            plt.xlim(origin, limit)
            plt.ylim(origin, limit)

        plot_index += 1
    
    if plot_file != "":
        fig.savefig(plot_file)
    return fig
    
def make_cluster_table(df: 'pandas.DataFrame', cluster_col: str='cluster_name', stat="median"):
    """Make a summary table of cluster median values and number of SNVs in each cluster.

    Args:
        df (pandas.DataFrame): a dataframe containing the CN-corrected SNVs and a column with the cluster assignment; 
                                all columns except the cluster assignment column should be sample columns
        cluster_col (str, optional): The name of the column with the cluster assignment. Defaults to 'cluster_name'.
    """
    counts = df[cluster_col].value_counts()
    if stat == "median":
        ccfs = df.groupby([cluster_col]).median().assign(num_snvs=df[cluster_col].value_counts())
    elif stat == "mean":
        ccfs = df.groupby([cluster_col]).mean().assign(num_snvs=df[cluster_col].value_counts())
    else:
        raise ValueError()
    ccfs['num_snvs'] = counts
    return(ccfs)

def save_df(df: 'pandas.DataFrame', filename: str, path: str=None):
    if path is not None:
        filename = os.path.join(path, filename)
    df.to_csv(filename)

def dpclust_workflow(dpinfo_files: list, inits: int=100, num_clusters: int=20, setlimits=True, stat="median", file_prefix: str="", limit=2.5): 
    dpinfo = load_data_with_locs(dpinfo_files)
    path = os.path.dirname(dpinfo_files[0])
    clusters = dpclust(dpinfo, inits=inits, num_clusters=num_clusters)
    dpinfo['cluster_name'] = clusters
    save_df(dpinfo, file_prefix + 'clusters_python.csv', path)
    plot_clusters(dpinfo, setlimits=setlimits, limit=limit)
    ccfs = make_cluster_table(dpinfo, stat=stat)
    save_df(ccfs, file_prefix + 'ccfs.csv', path) 
    return dpinfo, ccfs

def draw_annotated_clusters(df_final: 'pandas.DataFrame', ccfs_final: 'pandas.DataFrame', vcf: str=None, consequences: list=['missense_variant', 'stop_gained'], savefig: str="", anno_col: int=0, CSQ_col="CSQ="):
    """Draw individual SNVs coloured by their cluster assignment and annotate genes of interest

    Args:
        df_final (pandas.DataFrame): [description]
        ccfs_final (pandas.DataFrame): [description]
        vcf (str, optional): [description]. Defaults to None.
        consequences (list, optional): [description]. Defaults to ['missense_variant', 'stop_gained'].
    """
    if vcf is not None:
        vcf = pandas.read_csv(vcf, sep='\t', comment='#', header=None).iloc[:, :9]
        vcf.rename({0:'chr', 1:'end', 2:'ID', 3:'REF', 4:'ALT', 5:'QUAL', 6:'FILTER', 7:'INFO', 8:'FORMAT'}, axis=1, inplace=True)
        anno = pandas.DataFrame(vcf.apply(lambda x: re.split('\|', re.split(CSQ_col, x['INFO'])[1])[:anno_col+24], axis=1).tolist())
        anno.columns = ['Allele', 'Consequence', 'IMPACT', 'SYMBOL', 'Gene', 'Feature_type', 'Feature', 'BIOTYPE', 'EXON', 'INTRON', 
                        'HGVSc', 'HGVSp', 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 'Existing_variation', 
                        'ALLELE_NUM', 'DISTANCE', 'STRAND', 'FLAGS', 'PICK', 'VARIANT_CLASS']
        vcf_expanded = pandas.concat([vcf, anno], axis=1)
        vcf_expanded = vcf_expanded.set_index(['chr', 'end'])
        ccfs_final = ccfs_final.reset_index(drop=True)
        print(ccfs_final)
        df_final_filtered = df_final[df_final['cluster_name'].isin(ccfs_final['cluster_name'])]
        df_anno = df_final_filtered.join(vcf_expanded)
        print("Consequence table before sorting:\n", df_anno.groupby('Consequence').count()['SYMBOL'])
        import matplotlib.patches as patches
        import string
        cluster_names = ccfs_final['cluster_name'] # df_final_filtered['cluster_name'].unique()
        cluster_names_dict = dict(zip(cluster_names, range(len(cluster_names))))
        print(cluster_names_dict)
        df_anno['cluster_sno'] = df_anno.apply(lambda x: cluster_names_dict[x['cluster_name']], axis=1)
        df_anno['cluster_alphabet'] = df_anno.apply(lambda x: string.ascii_uppercase[int(x['cluster_sno'])], axis=1)
        df_anno = df_anno.sort_values(by='cluster_name', ascending=True).reset_index()
        df_anno['locus_sno'] = df_anno.index
        print("Consequence table:\n", df_anno.groupby('Consequence').count()['SYMBOL'])
        df_anno_melted = df_anno.melt(id_vars=['chr', 'start', 'end', 'cluster_name', 'cluster_sno', 'cluster_alphabet', 'locus_sno', 
                                                'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT',
                                                'Allele', 'Consequence', 'IMPACT',
                                                'SYMBOL', 'Gene', 'Feature_type', 'Feature', 'BIOTYPE', 'EXON',
                                                'INTRON', 'HGVSc', 'HGVSp', 'cDNA_position', 'CDS_position',
                                                'Protein_position', 'Amino_acids', 'Codons', 'Existing_variation',
                                                'ALLELE_NUM', 'DISTANCE', 'STRAND', 'FLAGS', 'PICK', 'VARIANT_CLASS'], 
                                    var_name='sample_name', value_name='ccf')
        df_anno_melted
        samples = df_anno_melted['sample_name'].unique()
        samples_dict = dict(zip(samples, range(len(samples))))
        df_anno_melted['sample_num'] = df_anno_melted.apply(lambda x: samples_dict[x['sample_name']], axis=1)
        vlines_data = df_anno_melted.groupby('cluster_alphabet').locus_sno.max()
        clusters_dict = df_anno_melted.groupby(['cluster_alphabet']).locus_sno.mean().to_dict()
        # clusters_dict
        df_with_consequences = df_anno_melted[df_anno_melted['Consequence'].isin(consequences)].drop_duplicates(subset='SYMBOL')
        goi_vlines = df_anno_melted[df_anno_melted['Consequence'].isin(consequences)]['locus_sno']
        fig, ax = plt.subplots(figsize=(26, 10))


        plt.xticks(ticks=list(clusters_dict.values()), labels=list(clusters_dict.keys()))
        plt.yticks(ticks=[x+0.25 for x in samples_dict.values()], labels=list(samples_dict.keys()))
        plt.ylim((-0.25, len(samples) + 0))
        plt.xlim((-1, max(df_anno_melted['locus_sno'])))
        
        ax2 = ax.twiny()
        ax2.set_xticks(df_with_consequences['locus_sno'])
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticklabels(df_with_consequences['SYMBOL'], rotation=45, ha='left')

        plt.vlines(vlines_data, -0.25, len(samples), zorder=4, linestyles='dashed', linewidth=0.45)
        custom_pal = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', 
                    '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', 
                    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
        # df_final.apply(lambda x: print(x.index), axis=1)
        df_anno_melted.apply(lambda x: ax.add_patch(patches.Rectangle((x['locus_sno'], x['sample_num'] + ((1 - x['ccf'])/4)), 
                                                                        width=1, height=x['ccf']/2, facecolor=custom_pal[x['cluster_sno']], 
                                                                        edgecolor="black", linewidth=0.1)), axis=1)
        # df_anno_melted.drop_duplicates(subset='SYMBOL').apply(lambda x: ax.text(x['locus_sno'], len(samples) + 1, x['SYMBOL'], ha='left', rotation=45) if x['Consequence'] in consequences else None, axis=1)
        # + numpy.random.uniform(-1, 1)
        # df_anno_melted.apply(lambda x: ax.add_patch(patches.Rectangle((x['locus_sno'], x['sample_num'] + ((1 - x['ccf'])/4)), 
        #                                                                 width=1, height=1, color="black", linewidth=0.1, zorder=6)) if x['Consequence'] in consequences else None, axis=1)
        plt.vlines(goi_vlines, len(samples) + 1, len(samples) + 1, zorder=5, linestyles='dashed', linewidth=1)

            # saving figure
        if savefig != "":
            fig.savefig(savefig)

        plt.show()
        return df_anno_melted

def draw_ccfs_on_chromosomes(df_final: 'pandas.DataFrame', ccfs_final: 'pandas.DataFrame', vcf: str=None, consequences: list=['missense_variant', 'stop_gained'], cytoband_file: str = None, savefig: str = "", only_for_chrs: list=None, variant_class: list=['SNV'], CSQ_col="ANN="):
    """Draw individual SNVs coloured by their cluster assignment and annotate genes of interest

    Args:
        df_final (pandas.DataFrame): [description]
        ccfs_final (pandas.DataFrame): [description]
        vcf (str, optional): [description]. Defaults to None.
        consequences (list, optional): [description]. Defaults to ['missense_variant', 'stop_gained'].
    """
    # create offset
    from natsort import natsorted
    if only_for_chrs is not None:
        df_final = df_final.reset_index()
        df_final = df_final[df_final['chr'].isin(only_for_chrs)].set_index(['chr', 'start', 'end'])
        # print(df_final)
    print("plotting only variant classes:" + " ".join(variant_class))

    # prepare cytoband data
    if cytoband_file is not None:
        cytoband = pandas.read_csv(cytoband_file, sep="\t", header=None)
        cytoband.columns = ['chr', 'start', 'end', 'name', 'stain']
        cytoband['chr'] = cytoband['chr'].str.replace("chr", "")
        if only_for_chrs is not None:
            cytoband = cytoband[cytoband['chr'].isin(only_for_chrs)]
        chr_size = cytoband.groupby('chr')['end'].max()
        print(chr_size)
        chr_size = chr_size.reindex(index=natsorted(chr_size.index)).to_frame(name='end')
        chr_offset = chr_size.end.cumsum().shift(fill_value=0)
        cytoband_offset = cytoband.set_index('chr').join(chr_offset.to_frame(name="offset"))
        cytoband_offset['start_offset'] = cytoband_offset['start'] + cytoband_offset['offset']
        cytoband_offset['end_offset'] = cytoband_offset['end'] + cytoband_offset['offset']
        stain_pal = {'gneg':(0, 0, 0), 'gpos25':(0.25, 0.25, 0.25), 'gpos50':(0.50, 0.50, 0.50), 'gpos75':(0.75, 0.75, 0.75), 
        'gpos100':(0.95, 0.95, 0.95), 'acen':(0.9, 0.1, 0.1), 'gvar':(1, 1, 1), 'stalk':(1, 1, 1)}
        cytoband_offset['col'] = cytoband_offset.apply(lambda x: stain_pal[x['stain']], axis=1)
        cytoband_offset = cytoband_offset.reset_index()

    if vcf is not None:
        vcf = pandas.read_csv(vcf, sep='\t', comment='#', header=None).iloc[:, :9]
        print(vcf)
        vcf.rename({0:'chr', 1:'end', 2:'ID', 3:'REF', 4:'ALT', 5:'QUAL', 6:'FILTER', 7:'INFO', 8:'FORMAT'}, axis=1, inplace=True)
        anno = pandas.DataFrame(vcf.apply(lambda x: re.split("\|", re.sub(".*" + CSQ_col, "", x['INFO']))[:24], axis=1).tolist())
        anno.columns = ['Allele', 'Consequence', 'IMPACT', 'SYMBOL', 'Gene', 'Feature_type', 'Feature', 'BIOTYPE', 'EXON', 'INTRON', 
                        'HGVSc', 'HGVSp', 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 'Existing_variation', 
                        'ALLELE_NUM', 'DISTANCE', 'STRAND', 'FLAGS', 'PICK', 'VARIANT_CLASS']
        if only_for_chrs is not None:
            # only_for_chrs = [x.replace("chr", "") for x in only_for_chrs]
            vcf = vcf[vcf['chr'].isin(only_for_chrs)]
            print(vcf.head())
        print(anno)
        vcf_expanded = pandas.concat([vcf, anno], axis=1)
        vcf_expanded = vcf_expanded.set_index(['chr', 'end'])
        print(vcf_expanded)
        ccfs_final = ccfs_final.reset_index(drop=True)
        print(ccfs_final)
        df_final_filtered = df_final[df_final['cluster_name'].isin(ccfs_final['cluster_name'])]
        df_anno = df_final_filtered.join(vcf_expanded)
        # print("Consequence table before sorting:\n", df_anno.groupby('Consequence').count()['SYMBOL'])
        import matplotlib.patches as patches
        import string
        cluster_names = ccfs_final['cluster_name'] # df_final_filtered['cluster_name'].unique()
        cluster_names_dict = dict(zip(cluster_names, range(len(cluster_names))))
        # print(cluster_names_dict)
        df_anno['cluster_sno'] = df_anno.apply(lambda x: cluster_names_dict[x['cluster_name']], axis=1)
        df_anno['cluster_alphabet'] = df_anno.apply(lambda x: string.ascii_uppercase[int(x['cluster_sno'])], axis=1)
        df_anno = df_anno.sort_values(by='cluster_name', ascending=True).reset_index()
        df_anno['locus_sno'] = df_anno.index
        df_anno = df_anno[df_anno['VARIANT_CLASS'].isin(variant_class)]
        print(df_anno)
        print("df_anno var calss")
        # print("Consequence table:\n", df_anno.groupby('Consequence').count()['SYMBOL'])
        df_anno_melted = df_anno.melt(id_vars=['chr', 'start', 'end', 'cluster_name', 'cluster_sno', 'cluster_alphabet', 'locus_sno', 
                                                'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT',
                                                'Allele', 'Consequence', 'IMPACT',
                                                'SYMBOL', 'Gene', 'Feature_type', 'Feature', 'BIOTYPE', 'EXON',
                                                'INTRON', 'HGVSc', 'HGVSp', 'cDNA_position', 'CDS_position',
                                                'Protein_position', 'Amino_acids', 'Codons', 'Existing_variation',
                                                'ALLELE_NUM', 'DISTANCE', 'STRAND', 'FLAGS', 'PICK', 'VARIANT_CLASS'], 
                                    var_name='sample_name', value_name='ccf')
        print(df_anno_melted)
        # prepare cna data
        df_anno_melted_offset = df_anno_melted.set_index('chr').join(chr_offset.to_frame(name="offset"))
        print(df_anno_melted_offset.offset)
        df_anno_melted_offset['start_offset'] = df_anno_melted_offset['start'] + df_anno_melted_offset['offset']
        print(df_anno_melted_offset)
        
        samples = df_anno_melted_offset['sample_name'].unique()
        samples_dict = dict(zip(samples, range(len(samples))))
        print(samples_dict)
        df_anno_melted_offset['sample_num'] = df_anno_melted_offset.apply(lambda x: samples_dict[x['sample_name']], axis=1)
        vlines_data = df_anno_melted_offset.groupby('cluster_alphabet').locus_sno.max()
        clusters_dict = df_anno_melted_offset.groupby(['cluster_alphabet']).locus_sno.mean().to_dict()
        # clusters_dict
        df_with_consequences = df_anno_melted_offset[df_anno_melted_offset['Consequence'].isin(consequences)].drop_duplicates(subset='SYMBOL')
        goi_vlines = df_anno_melted_offset[df_anno_melted_offset['Consequence'].isin(consequences)]['locus_sno']
        fig, ax = plt.subplots(figsize=(26, 10))


        # prepare labels
        midchr = (cytoband_offset.groupby('chr')['end'].max()/2).astype('int')
        midchr = pandas.concat([cytoband_offset.groupby('chr')['offset'].max().to_frame(name='offset'), midchr.to_frame(name='midchr')], axis=1)
        midchr['mid_offset'] = midchr['offset'] + midchr['midchr']
        plt.xticks(ticks=midchr['mid_offset'], labels=midchr.index)
        
        plt.yticks(ticks=[x+0.25 for x in samples_dict.values()], labels=list(samples_dict.keys()))
        plt.ylim((-0.25, len(samples) + 2))
        ax.set_xlim((cytoband_offset['start_offset'].min(), cytoband_offset['end_offset'].max()))
        print("xbound:", ax.get_xlim())
        # ax2 = ax.twiny()
        # ax2.set_xticks(df_with_consequences['locus_sno'])
        # ax2.set_xlim(ax.get_xlim())
        # ax2.set_xticklabels(df_with_consequences['SYMBOL'], rotation=45, ha='left')
        vlines_data = cytoband_offset.groupby('chr').max('end_offset')['end_offset']
        vlines_data = vlines_data.reindex(index=natsorted(vlines_data.index))
        plt.vlines(vlines_data, 0, len(samples)+2, zorder=4, linestyles='dashed', linewidth=0.45)
        custom_pal = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', 
                    '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', 
                    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
        # df_final.apply(lambda x: print(x.index), axis=1)
        if cytoband_file is not None:
            cytoband_offset.apply(lambda x: ax.add_patch(patches.Rectangle((x['start_offset'], len(samples) + 1), width=x['end_offset'] - x['start_offset'], height=0.25, color=x['col'])), axis=1)

        df_anno_melted_offset.apply(lambda x: ax.add_patch(patches.Rectangle((x['start_offset'], x['sample_num'] + ((1 - x['ccf'])/4)), 
                                                                        width=1, height=x['ccf']/2, facecolor=custom_pal[x['cluster_sno']], 
                                                                        edgecolor=custom_pal[x['cluster_sno']], linewidth=1)), axis=1)
        # df_anno_melted.drop_duplicates(subset='SYMBOL').apply(lambda x: ax.text(x['locus_sno'], len(samples) + 1, x['SYMBOL'], ha='left', rotation=45) if x['Consequence'] in consequences else None, axis=1)
        # + numpy.random.uniform(-1, 1)
        # df_anno_melted.apply(lambda x: ax.add_patch(patches.Rectangle((x['locus_sno'], x['sample_num'] + ((1 - x['ccf'])/4)), 
        #                                                                 width=1, height=1, color="black", linewidth=0.1, zorder=6)) if x['Consequence'] in consequences else None, axis=1)
        plt.vlines(goi_vlines, len(samples) + 1, len(samples) + 1, zorder=5, linestyles='dashed', linewidth=1)

            # saving figure
        if savefig != "":
            fig.savefig(savefig)

        plt.show()
        return df_anno_melted_offset


