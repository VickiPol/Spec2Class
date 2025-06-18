
import pandas as pd
import sys
from prediction_vectors_functions import get_pred_vecs
from binning_functions import binning_df
from multiclass_prediction_functions import svm_pred
import configparser
from utility_functions import read_df_and_format_mz_intensity_arrays


def main(chemclass_list, input_df, ms1_tol,bin_width, start_value, end_value, output_name,output_dir,net_path,
         net_dir, dropout_conv, dropout_linear, binary_models_dir, svm_model_path ,params):
    """

    :param chemclass_list: (list) list of all predicted classes
    :param input_df: (DataFrame) with the input set
    :param ms1_tol: (int) relevant if ExactMass exists. Fragments with mass higher than ms1_tol+ExactMass will be dropped
    :param bin_width: (int) the mass fragments resolution, set to 0.1 Da for the given model
    :param start_value: minimal fragment's mass, 50 Da
    :param end_value:  maximal fragment's mass, 550 Da
    :param output_name: (str)
    :param output_dir: (str) output directory
    :param net_path: (str) path to the NN class
    :param net_dir: (str) NN directory
    :param dropout_conv: (float) 0.95 dropout ratio (used to initialize the NN but not used in inference)
    :param dropout_linear: (float) 0.95 dropout ratio (used to initialize the NN but not used in inference)
    :param binary_models_dir: (str) the directory to where all the trained models files are saved
    :param svm_model_path: (str) the directory to where the trained SVM model is saved
    :param params: (dict) parameters for inference in batches
    :return: (DataFrame) with all final multiclass prediction
    """

    binned_df = binning_df(input_df, bin_width, start_value, end_value, output_name, ms1_tol)
    # TODO: check if get_pred_veds needs update. Can it deal with positive and negative models?
    pred_vecs_df = get_pred_vecs(chemclass_list, output_name, binned_df, output_dir, end_value, start_value, bin_width,
                                 net_path, net_dir, dropout_conv, dropout_linear, binary_models_dir, params)
    # TODO: add the p1-p2 value to the output
    final_pred_df = svm_pred(svm_model_path, output_dir, output_name, pred_vecs_df, chemclass_list)

    return final_pred_df



if __name__ == "__main__":

    config_file_path = rf"{sys.argv[1]}"
    input_path = rf"{sys.argv[2]}"
    output_dir = rf"{sys.argv[3]}"
    output_name = sys.argv[4]

    config_obj = configparser.ConfigParser()
    config_obj.read(config_file_path)
    binary_models_dir = config_obj['paths']['binary_models_dir']
    net_path = config_obj['paths']['net_path']
    net_dir = config_obj['paths']['net_dir']
    svm_model_path = config_obj['paths']['svm_model_path']
    end_value = int(config_obj['model']['end_value'])  # 550
    start_value = int(config_obj['model']['start_value'])  # 50
    bin_width = float(config_obj['model']['bin_width'])  # 0.1
    dropout_conv = float(config_obj['model']['dropout_conv'])  # 0.95
    dropout_linear = float(config_obj['model']['dropout_linear'])  # 0.95
    ms1_tol = float(config_obj['model']['ms1_tolerance'])
    chemclass_list = config_obj['chemclass_lists']['superclass_list'].split('\n')
    batch_size = int(config_obj['dataloader']['batch_size'])
    num_workers = int(config_obj['dataloader']['num_workers'])
    params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_workers}
    input_df = read_df_and_format_mz_intensity_arrays(input_path)
    results_df = main(chemclass_list, input_df, ms1_tol,bin_width, start_value, end_value, output_name,output_dir,net_path,
         net_dir, dropout_conv, dropout_linear, binary_models_dir, svm_model_path ,params)














