from matchms.importing import load_from_mgf
import pandas as pd
import numpy as np
import os


def save_df(filename, file_dir, df):
    """
    function that saves a data frame in .tsv, .csv, .pkl formats
    :param filename: (str) the name of saved file
    :param file_dir: (str) file's directory
    :param df: (pd.Dataframe) data frame meant for saving
    :return: none
    """

    save_dir = os.path.join(file_dir,filename)
    df.to_pickle(save_dir + '.pkl', compression = None)
    df.to_csv(save_dir+'.csv')
    df.to_csv(save_dir+'.tsv', sep = '\t')
    print(f"Saved {filename} ")

def mgf_to_df(mgf_file_path):
    """
    A function that parses an .mgf file which contains multiple spectra
    It uses functions from matchms library

    :param mgf_file_path: (str) path to the .msp file
    :return: df: (pd.Dataframe) a data frame where each line is a spectrum
    """

    spectra_from_path = list(load_from_mgf(mgf_file_path)) #creates a list of spectrum objects
    metadata_dict = spectra_from_path[0].metadata  #to get the format metadata keys
    metadata_df = pd.DataFrame(columns = list(metadata_dict.keys()) + ['mz','intensities'], dtype = 'object')
    for i in range(len(spectra_from_path)):
        metadata_dict_i = spectra_from_path[i].metadata
        for key in metadata_dict_i.keys():
            metadata_df.loc[i, key] = metadata_dict_i[key]
        mz_arr = spectra_from_path[i]._peaks.mz
        intensities_arr = spectra_from_path[i]._peaks.intensities
        metadata_df.loc[i, 'mz'] = mz_arr
        metadata_df.loc[i, 'intensities'] = intensities_arr

    return metadata_df


def msp_to_df(file_path, min_mass):
    """
    This function parses an .msp file that contains only a single spectrum

    :param file_path: (str) path to the .msp file
    :param min_mass:(float) the minimal mass to charge ratio (m/z) of a fragment to be included.
    :return: (pd.Dataframe) a single row dataframe
    """
    # This function parses a whole msp file and returns a df
    mz = list()
    intensity = list()
    file = open(file_path, 'r')
    file_txt = file.read()
    i = 0
    metadata_dict = {}
    txt_rows = file_txt.split('\n')
    for row in txt_rows:
        row = row.strip()
        if row == '':
            continue
        if ':' in row:
            key = row.split(':')[0]
            value = row.split(':')[1]
            metadata_dict[key] = value
        elif '\t' in row:
            if float(row.split('\t')[0]) > min_mass:
                mz = mz + [row.split('\t')[0]]
                try:
                    intensity = intensity + [row.split('\t')[1]]
                except IndexError:
                    continue
        else:
            try:
                float(row.split(' ')[0])
                if float(row.split(' ')[0]) > min_mass:
                    mz = mz + [row.split(' ')[0]]
                    try:
                        intensity = intensity + [row.split(' ')[1]]
                    except IndexError:
                        continue
            except ValueError:
                print(f'problematic row : {row}')
                continue
            else:
                continue
    mz_arr = np.asarray(mz, dtype=float)
    intensities_arr = np.asarray(intensity, dtype=float)
    if i == 0:
        df = pd.DataFrame(columns = list(metadata_dict.keys()) + ['mz','Intensity'], dtype = 'object')
    for key in metadata_dict.keys():
        df.loc[i, key] = metadata_dict[key].strip()
    df.loc[i, 'mz'] = mz_arr
    df.loc[i, 'Intensity'] = intensities_arr

    return df


def merged_msp_to_df(file_path, min_mass):
    """
    .msp files can contain a single spectrum but often times they contain several spectra in a single file.
    This function parses a file that contains several spectra
    :param file_path: (str) the path to the .msp file
    :param min_mass: (float) the minimal mass to charge ratio (m/z) of a fragment to be included.
    :return: df: (pd.Dataframe) a data frame where each line is a spectrum
    """

    file = open(file_path, 'r')
    file_txt = file.read()
    txt_list = file_txt.split('\n\n')
    for i in range(len(txt_list)):
        print(i)
        mz = list()
        intensity = list()
        metadata_dict = {}
        txt_rows = txt_list[i].split('\n')
        for row in txt_rows:
            row = row.strip()
            if row == '':
                continue
            if ':' in row:
                key = row.split(':')[0].strip()
                value = row.split(':')[1].strip()
                metadata_dict[key] = value
            elif '\t' in row:
                if float(row.split('\t')[0]) > min_mass:
                    mz = mz + [row.split('\t')[0]]
                    try:
                        intensity = intensity + [row.split('\t')[1]]
                    except IndexError:
                        continue
            else:
                if float(row.split(' ')[0]) > min_mass:
                    mz = mz + [row.split(' ')[0]]
                    try:
                        intensity = intensity + [row.split(' ')[1]]
                    except IndexError:
                        continue
                else:
                    continue
        mz_arr = np.asarray(mz, dtype=float)
        intensities_arr = np.asarray(intensity, dtype=float)
        if i == 0:
            df = pd.DataFrame(columns = list(metadata_dict.keys()) + ['mz','Intensity'], dtype = 'object')
        for key in metadata_dict.keys():
            df.loc[i, key] = metadata_dict[key]
        df.at[i, 'mz'] = mz_arr
        df.at[i, 'Intensity'] = intensities_arr

    return df

