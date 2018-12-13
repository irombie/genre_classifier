### Decision Tree Code (Nikhil Gowda)
## Usage
Can be run by simply running the python script by extracting features. Feature extraction code is commented within the single python file. Dataset  (~1.2gb). Just specify path of each genre folder, the script should do the rest to extract all features, specifically tempo, MFCC, and Chroma CQT
## Step 1
Download dataset http://marsyasweb.appspot.com/download/data_sets/ 
Make sure directory is specified in script that generated maindataset.txt/csv

## Step 2
Ensure to install libROSA library for raw audio feature extraction ttps://librosa.github.io/librosa/feature.html

## Step 3
Python program can be first run with feature extraction then by specifying the correct path in decision tree python script, the problem can be run by "python3 multi_class_tree.py" 

## References/Extensions
Got much guidance from machinelearningmastery.com on how to build a decision tree. In addition, this code may be able to be extended to include various different datasets if parameters are changed to be more general. Future goals also may include to properly engage with 2d Array features next to single scalar elements instead of the requirement of there being same dimensional data. 

## Collaborators
I'd like to thank my wonderful partners of Irem and Chen who guided me through this project. 
