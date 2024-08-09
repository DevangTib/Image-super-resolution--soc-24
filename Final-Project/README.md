This is an implementation of the ESPCN(Efficient Sub Pixel Convolution Neural Network) model for image super resolution. (https://arxiv.org/pdf/1609.05158).  

The structure of the CNN can be seen in the /thumbnails.  

Training:  

The Set5 (https://drive.google.com/file/d/1YfWiI4mO1yyRYHxnjXbNucCVuU52cpvd/view?usp=sharing) and 91-image (https://drive.google.com/file/d/1XYjEmgAmtGOpXu6ep8931o4sfWI9NAYC/view?usp=sharing) dataset from DIV2K coverted to hdf5 can be downloaded from given links. Else, prepare.py can be used for creating custom dataset.  

For training, run the following command:  

python3 train.py --train-file "(path to dataset)/91-image_x3.h5"  --eval-file "(path to dataset)/Set5_x3.h5" --outputs-dir "(Path for storing output)/outputs" --scale 3 --lr 1e-3 --batch-size 16 --num-epochs 200 --num-workers 8 --seed 123 (parameters can be adjusted)  


Test:  

Pretrained weights can be downloaded from here (https://drive.google.com/file/d/1lsJs5CFE9GFfBpob1irkas1TptfFQdz3/view?usp=sharing). Run the following command to test:  

python3 test.py --weights-file "(Path to dataset)/espcn_x3.pth" --image-file "data/butterfly_GT.bmp" --scale 3
(The results are stored in the /data folder)
