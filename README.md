Spec2Class
==========
Predicting metabolite chemical class out of LC-MS/MS spectrum
-------------------------------------------------------------
Developed in Aharoni and Zeevi labs at the Weizmann Institute of Science
--------------------------------------------------------------------------
Table of contents
-----------------

a. Abstract  
b. Model description  
c. Input description  
d. Output description  
e. How to use?  
  

a. Abstract
-----------

Mass spectrometry is commonly used in studying metabolism and natural products, but typically requires domain-specific skill and experience to analyze. Existing computational tools for metabolomics analysis mostly rely on comparison to reference spectral libraries for metabolite identification, limiting the annotation of metabolites for which reference spectra do not exist.  
We developed Spec2Class, a deep-learning algorithm for the classification of plant secondary metabolites from high resolution LC-MS/MS spectra. We used a unique in-house spectral library of ~8000 plant metabolite chemical standards, alongside publicly available data, to train Spec2Class to classify LC-MS/MS spectra to tens of common plant secondary metabolite classes. Tested on held out sets, our algorithm achieved an accuracy of 73%, and an average binary classification auROC of 0.94.

b. Model description
--------------------

Spec2Class is an ensemble classification model built out of 43 binary classifiers that serve as base classifiers. Each binary classifier is a neural net model built out of two convolutional layers, followed by three fully connected linear layers. For each binned spectrum 43 binary predictions are generated and concatenated to a single vector that serves as an input to an SVM model that provides the final multiclass prediction of the chemical class.  
![image](https://github.com/VickiPol/Spec2Class/blob/main/architecture_v1.png)

c. Input description
--------------------

The model’s input should be in a tabular format and saved in a .pkl (pickle) format. Each row in the input dataframe should represent one **positive** LC-MS/MS spectrum.  
The dataframe can contain different metadata columns, but should contain the following mandatory columns:  
1.**‘mz’** – array of m/z values for each row  
2.**‘Intensity’** – array of corresponding relative intensities  
3.**‘DB.’** – spectrum identifier  
If the information about the exact mass of the parent ion exsists, name this field 'ExactMass'. The parent's ion m/z is used only in spectrum the binning stage, before the first step of prediction.  
Fragments that have m/z ratio higher than parent ion mass + 0.01 Da are dropped. If the information about the parent ion m/z is missing, then all the fragments will between 50 and 550 Da will be included.  
**See input example: mona\_100\_spec.pkl**  
MS/MS data has different output formats. Among mostr frequent are .mgf and .msp. These formats are usually similar among different data processing platforms but not always identical.
in **input_parsing_functions.py** you can find parsing functions that take as .mgf or .msp files as input and output a datframe that can be used with Spec2Class. These functions served us for parsing, but please take into consideration that similar file types that come from other sources might require small corrections for the given functions. 

d. Output description
---------------------

The output is a tabular file in three formats: .pkl,.tsv,.csv  
The output will contain the following columns:  
**DB.** – spectrum identifier  
**final pred** – chemical class prediction  
**estimated top2 pred** – chemical class prediction with the 2nd highest probability  
**estimated top3 pred** – chemical class prediction with the 3rd highest  
**probabilities** – array of the top 3 probability values  

e. How to use?
--------------

Please find here user's manual in pdf format:\
https://github.com/VickiPol/Spec2Class/blob/main/Spec2Class_manual.pdf \
### 1\. Create spect2class conda environment with the provided file spec2class_env.yml  
  
### 2\. Download the trained models  
2.1 Download and save all the binary models (43 models) from hugging face hub: https://huggingface.co/VickiPol/binary_models  
  
2.2 Download and save the SVM model from hugging face hub: https://huggingface.co/VickiPol/SVM/model  
  
### 3\. Edit the paths in the config file config_spec2class.ini:  
  
For example:  
\[paths\]  
#The path to the trained SVM model\
`svm_model_path = \Spec2Class\SVM_model\spec2class_trained_svm.sav` \
#The path to the directory where the 43 binary models are saved \
`binary_models_dir = \Spec2Class\binary_models\binary`   
#The path to the Neural Net class   
`net_path = \Spec2Class\neural_net.py`    
#The path to the directory Neural Net class   
`net_dir = \Spec2Class\classes`   
  
### 4\. Run the model with the given input example:  
  
`python Spec2Class.py [config_file_path] [input_path] [output_directory] [output_name]`  
  
config file path is the path to config\_spec2class.ini  
input\_path is the path to mona\_100\_spec.pkl  
output directory and name is straight forward :)  
  
### 5\. If everything worked your good to go with your own data! else contact us :)  
  
### 6\. It is possible and even recomended to run Spec2Class on GPU. The inference will be much faster. If 'cuda' is available GPU will ne used automatically.
