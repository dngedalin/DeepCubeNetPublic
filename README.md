# DeepCubeNet

DeepCubeNet is a deep learning architecture for reconstruction of Compressive sensed Hyperspectral images.

In order to create training and validation patches data using the create_data.py script, make sure to use the correct configuration.
Paper configuration train, validation and test sets are given in the corresponding .txt files in src/config.
Notice that while train and validation are calculated in patches, the test is over the whole full cubes.
To create test data use the create_data script with patch_flag=False and create new path for saving.

For results see the paper DeepCubeNet ( To be published in OpticsExpress).

![Alt text](https://github.com/dngedalin/DeepCubeNet/blob/master/assets/paper_val_loss.PNG "paper_val_loss")
![Alt text](https://github.com/dngedalin/DeepCubeNet/blob/master/assets/paper_val_psnr.PNG "paper_val_psnr")

