# Video Classification and Anomaly Detection
This is our deep learning project of group 12 

Project aim: detect which video has anomalies and where it happens

Dataset: UCF-Crime real-world surveillance videos, you can download or read more about it here: https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0

Full-dataset is used for Anomaly Detection while only a small part of it is used for classification, small part can be referred here: https://www.kaggle.com/datasets/mission-ai/crimeucfdataset  

## Feature Extraction for Anomaly Detection

we use I3D for Spatial-Temporal Feature Extraction with output is rgb and optical flow numpy file, read more and run in file `feature-extraction.ipynb`
