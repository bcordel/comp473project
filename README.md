# Handwritten Chinese Character Recognition (HCCR) 
This code aims to perform HCCR on the ICDAR-2013 dataset. The code here was written by Briac Cordelle, Samuel something and Andrew Robinson. 

The following files are defined: 
hcc2.py - trains a neural network using data found in the ./data/ directory
gnt.py - reads and converts .gnt byte pictures downloaded from the ICDAR-2013 dataset to .jpg images to be used in the network
directmap.py - attempts to decompose .jpg images into direction decomposed maps 
test.py - tests the model based on specific unit-tests


## How to use it
You can find a trained model here: 
https://drive.google.com/file/d/1-RojVjZz-NXiAGIGTjXej8BGTfo_iaCc/view?usp=sharing
This model achieves a 61% accuracy on the train set. Use the test.py file to test it. 

If you would like to train a model yourself, download the files in /data/gnt/. 
Then convert them to jpg files using the code in main of gnt.py
Make sure the appropriate directories are being used in hcc2.py 
Run main in hcc2.py to the desired number of epochs.
Note that hcc2.py uses 'cuda' by default, if you would like to change this, be sure to change the device variable in main.
Currently the directmap.py does not work as hoped so the preprocess step in hcc2 performs some basic preprocessing then calculates the magnitude of the gradient by the sobel operator for edge detection. 

