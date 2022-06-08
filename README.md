# Music Genre Classification Using CNNs (Irem Ergun)
## For the CNN: 
### The accuracy achieved is 70%, which is 7 times better than random assignment. This also outperforms the other two algorithms we implemented (kNN and decision trees) significantly. 
- cnn.py is for constructing the convolutional neural network.
- feature.py is for preprocessing for feature extraction
- mini.py is the code we run to see results, which we have to run by specifying the data folder
A toy dataset is provided in toy_data folder, just extract the archive and run
To run, you can use the Makefile: just type "make run". If you do not change the location of the toy dataset, it should work. 
Alternatively, you can just run by typing "python mini.py --dataset_folder_path=$(DATASET_PATH)" 
Due to spatial constraints, I could just provide you a very small portion of the dataset. The code does not give a meaningful output for that, but it shows that code is working. If you want to actually run the code, please email me (iremlergun@gmail.com) so that I can give you the whole dataset. (For some reason, I cannot find the original dataset online but I can provide you with the chromagram of the songs as the dataset, which is what we used to train our classifier.)

# For kNN:
Usage:
	Put CS235_kNN.py and /genres under the same folder and run
	./python CS235_kNN.py

# Decision Tree Code (Nikhil Gowda)
## Usage
Can be run by simply running the python script by extracting features. Feature extraction code is commented within the single python file. Dataset  (~1.2gb). Just specify path of each genre folder, the script should do the rest to extract all features, specifically tempo, MFCC, and Chroma CQT
## Step 1
Get the dataset

## Step 2
Ensure to install libROSA library for raw audio feature extraction https://librosa.github.io/librosa/feature.html

## Step 3
Python program can be first run with feature extraction then by specifying the correct path in decision tree python script, the problem can be run by "python3 multi_class_tree.py" 

## References/Extensions
Got much guidance from machinelearningmastery.com on how to build a decision tree. In addition, this code may be able to be extended to include various different datasets if parameters are changed to be more general. Future goals also may include to properly engage with 2d Array features next to single scalar elements instead of the requirement of there being same dimensional data. 

## Collaborators
I'd like to thank my wonderful partners of Irem and Chen who guided me through this project. 
