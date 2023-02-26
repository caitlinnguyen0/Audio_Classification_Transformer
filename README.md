# Audio_Classification_Transformer

## Setup 
Transformer model was trained on GoogleColab running python 3.8.10.

Samples should be downloaded and added to the Google Drive account associated with the account running Colab. In a new notebook create a folder called "Transformer_training_audio" within the content directory.

## Downloading Training and Testing Data
8,732 audio samples can be downloaded from [UrbanSound8K](https://urbansounddataset.weebly.com/download-urbansound8k.html).

Audio augmentations were applied to the samples (which can be downloaded here [Google Drive](https://drive.google.com/file/d/1B6sy2_Llh5zAQ3yHBeQw93udIMOgSFfe/view?usp=sharing)) to total 69,856 training and testing data as represented in [new.csv](new.csv). 

Data breakdown is as follows: 2992 gunshot samples, 66864 non-gunshot samples

augment, class, class_id, file, fold 
