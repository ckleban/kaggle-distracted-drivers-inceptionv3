# kaggle-distracted-drivers-inceptionv3
Starter script for Kaggle's distracted driver comp. Tensor Flow + transfer learning + inception v3 pre-trainned model

# Kaggle Comp
These files are for the distracted driver comp on Kaggle, found here: https://www.kaggle.com/c/state-farm-distracted-driver-detection/

# Summary

These files let you take the pre-trained inception v3 image reconigtion model, re-train it on the Kaggle data and perform some preditions. These are python scripts that use the tensorflow ML library. This is meant as a starter script of sorts for others to try to do the same thing. 

Using pretrainned models can reduce training time and can be done without a CPU. 

Using these scripts, you get a Kaggle leader board score of about 1.2 (just need to play with the hyper-params. Right now, there is some over-fitting that is happening that needs to be solved in order to reduce the error rate and improve your LB score. 

# Background reading:

* Basic image reconition in tensorflow: https://www.tensorflow.org/versions/r0.9/tutorials/image_recognition/index.html
* Retraining walk through in tensorflow: https://www.tensorflow.org/versions/r0.9/how_tos/image_retraining/index.html


# Prereqs

* Python scripts: Python, tensorflow, pickle, panda, etc 
* (Optional) Image pre-processsing: ImageMagick
* 
# Script Usage

* (Optional) - Increase the training data set size by doing image distortions. (To try to solve overfitting).

```
bash /data/BitBucket/distracted-drivers/process_image.sh 
```

* Run the retrain.py script to perform the retraining and create your new model
```
python ./retrain.py --image_dir='/data/new-images/input/processed/train/' --output_graph='/data/drivers/output_graph_june1-test.pb' --output_labels='/data/drivers/output_labels_june1-test.txt' --how_many_training_steps=50000             --bottleneck_dir='/data/bottlenecks-processed' --learning_rate=0.003 --train_batch_size=64 --usingProcessed=True 
```

Note: if you are not using the process_image.sh script to create more training data, change --usingProcessed to False

* Change the model-use-pretrained-no-skflow.py file's variables to match the files names of your model and label files:

```
pbfilename = 'output_graph_june1-test.pb'
pblabelfilename='output_labels_june1-test.txt' 
uidfilename='output_labels_june1-test.txt' 
```

* Note: I hard coded the label output order in model-use-pretrained-no-skflow.py. If your --output_labels file has a different order than the below line in  model-use-pretrained-no-skflow.py, you should change the below line to match the order:

```
 result1 = pd.DataFrame(predictions, columns=['c9', 'c8', 'c3', 'c2', 'c1', 'c0', 'c7', 'c6', 'c5', 'c4'])
```


* Run the model-use-pretrained-no-skflow.py script to make your predictions. 

```
python ../BitBucket/distracted-drivers/model-use-pretrained-no-skflow.py --model_dir='/data/drivers'
```
