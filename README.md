<h1>Spec2Class</h1>
<h2>Predicting metabolite chemical class out of LC-MS/MS spectrum</h2>

<h2>Table of contents</h2>
1. Abstract<br>
2. Model description<br>
3. Input description<br>
4. Output description<br>
5. How to use?<br>
<br>
<h2>Abstract</h2>
Mass spectrometry is commonly used in studying metabolism and natural products, but typically requires domain-specific skill and experience to analyze. Existing computational tools for metabolomics analysis mostly rely on comparison to reference spectral libraries for metabolite identification, limiting the annotation of metabolites for which reference spectra do not exist. <br>
We developed Spec2Class, a deep-learning algorithm for the classification of plant secondary metabolites from high resolution LC-MS/MS spectra. We used a unique in-house spectral library of ~8000 plant metabolite chemical standards, alongside publicly available data, to train Spec2Class to classify LC-MS/MS spectra to tens of common plant secondary metabolite classes. Tested on held out sets, our algorithm achieved an accuracy of 73%, and an average binary classification auROC of 0.94.
<h2>Model description</h2>
Spec2Class is an ensemble classification model built out of 43 binary classifiers that serve as base classifiers. Each binary classifier is a neural net model built out of two convolutional layers, followed by three fully connected linear layers. For each binned spectrum 43 binary predictions are generated and concatenated to a single vector that serves as an input to an SVM model that provides the final multiclass prediction of the chemical class.<br>
![image](https://github.com/AharoniLab/Spec2Class/blob/main/architecture_v1.png, raw = true)
<h2>Input description</h2>
The model’s input should be in a tabular format and saved in a .pkl  (pickle) format. Each row in the input dataframe should represent one **positive** LC-MS/MS spectrum.<br>
The dataframe can contain different metadata columns, but should contain the following mandatory columns:<br>
1.‘mz’ – array of m/z values for each row<br>
2.‘Intensity’ – array of corresponding relative intensities <br>
3.‘DB.’ – spectrum identifier <br>
If information about the exact mass of the parent ion exsists, name this field 'ExactMass'. <br>
See example: mona_100_spec.pkl<br>
<h2>Output description</h2>
The output is a tabular file in three formats: .pkl,.tsv,.csv<br>
The output will contain the following columns: <br>
DB. – spectrum identifier <br>
final pred – chemical class prediction <br>
estimated top2 pred – chemical class prediction with the 2nd highest probability<br>
estimated top3 pred – chemical class prediction with the 3rd highest <br>
probabilities – array of the top 3 probability values <br>
<h2>How to use?</h2>
Please find here user's manual in pdf format: 
1. Create spect2class conda environment with the proveided file spec2class_env.yml<br>
<br>
2. Download the trained models<br>
  2.1 Download and save all the binary models (43 models) from hugging face hub: https://huggingface.co/VickiPol/binary_models<br>
  <br>
  2.2 Download and save the SVM model from hugging face hub: https://huggingface.co/VickiPol/SVM_model<br>
  <br>
3. Edit the paths in the config file config_spec2class.ini:<br>
<br>
For example:<br>
[paths]<br>
#The path to the trained SVM model<br>
svm_model_path = \Spec2Class\SVM_model\spec2class_trained_svm.sav<br>
#The path to the directory where the 43 binary models are saved<br>
binary_models_dir = \Spec2Class\binary_models\binary<br>
#The path to the Neural Net class<br>
net_path = \Spec2Class\new_model_b550.py<br>
#The path to the directory Neural Net class<br>
net_dir = \Spec2Class\new_models<br>
<br>
4. Run the model with the given input example:<br>
<br>
python Spec2Class.py config_file_path input_path output_directory output_name <br>
<br>
config file path is the path to config_spec2class.ini<br>
input_path is the path to mona_100_spec.pkl<br>
output directory and name is straight forward :)<br>
<br>
5. If everything worked your good to go with your own data! else contact us :)<br>
<br>
6. It is possible and even recomended to run Spec2Class on GPU. The inference will be much faster. If 'cuda' is available GPU will ne used automatically.<br>
<br>
