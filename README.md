# Video Classification and Anomaly Detection
This is our deep learning project of group 12 

Project aim: detect which video has anomalies and where it happens

Dataset: UCF-Crime real-world surveillance videos, you can download or read more about it here: https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0

Full-dataset is used for Anomaly Detection while only a small part of it is used for classification, small part can be referred here: https://www.kaggle.com/datasets/mission-ai/crimeucfdataset  

## Feature Extraction for Anomaly Detection

we use I3D for Spatial-Temporal 32 segments feature extraction with output is rgb and optical flow numpy file

- Input
The inputs are paths to video files. Paths can be passed as a list of paths or as a text file formatted with a single path per line.

- Output
Output is defined by the on_extraction argument; by default it prints the features to the command line. Possible values of output are ['print', 'save_numpy', 'save_pickle']. save options save the features in the output_path folder with the same name as the input video file but with the .npy or .pkl extension.

Read more and run in file `feature-extraction.ipynb`

## Training for Anomaly Detection 

After finishing extract feature for full-dataset, we will use it for training, it already has been uploaded to Kaggle: https://www.kaggle.com/datasets/kanishkarav/ucf-crime-video-dataset

File `model.ipynb` will train then save weight model

## Some Results 

### Classification

### Anomaly Detection 

<td><img alt="" src="./media_images_ROC Curve.png" />

## Demo

File `DemoVidA.ipynb`

<table>
  <tr>
    <td><img alt="" src="./Arrest002gif.gif" /></td> <td><img alt="" src="./Arrest002_x264_result.png" height="280" width="400" />
  <tr>
</table>
