## Deep3D GCN-ResNet ##


## Getting Started
### Testing Requirements ###

- Reconstructions can be done on both Windows and Linux. However, we suggest running on Linux because the rendering process is only supported on Linux.
- Python 3.6 (numpy, scipy, pillow, argparse).
- Tensorflow 1.12.
- [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model). 
- [Expression Basis (transferred from Facewarehouse by Guo et al.)](https://github.com/Juyong/3DFace). The original BFM09 model does not handle expression variations so extra expression basis are needed. 
- [tf mesh renderer](https://github.com/google/tf_mesh_renderer).  We use the library to render reconstruction images. Follow the instruction of tf mesh render to install it using Bazel. Note that the rendering tool can only be used on Linux.

### Testing with pre-trained network ###

1. Clone the repository 

```bash
git clone https://github.com/Microsoft/Deep3DFaceReconstruction
cd Deep3DFaceReconstruction
```

2. Download the Basel Face Model. Due to the license agreement of Basel Face Model, you have to download the BFM09 model after submitting an application on its [home page](https://drive.google.com/file/d/1xNbklNSejHnYDT-w3E9CcHhsY2wo_MiL/view?usp=sharing). After getting the access to BFM data, download "01_MorphableModel.mat" and put it into ./BFM subfolder.

3. Download the Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace) You can find a link named "CoarseData" in the first row of Introduction part in their repository. Download and unzip the Coarse_Dataset.zip. Put "Exp_Pca.bin" into ./BFM subfolder. The expression basis are constructed using [Facewarehouse](kunzhou.net/zjugaps/facewarehouse/) data and transferred to BFM topology.

4. Compile [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer) with Bazel. We recommend using its [older version](https://github.com/google/tf_mesh_renderer/tree/ba27ea1798f6ee8d03ddbc52f42ab4241f9328bb) because we find its latest version unstable during our training process. If the library is compiled correctly, there should be a file named "rasterize_triangles_kernel.so". Put the .so file into ./renderer folder. For convenience, we provide a [pre-compiled file](https://drive.google.com/file/d/1VUtJPdg0UiJkKWxkACs8ZTf5L7Y4P9Wj/view?usp=sharing) of the library under tensorflow 1.12. Download the file and put it in ./renderer.

5. Download the pre-trained [reconstruction network](https://drive.google.com/file/d/176LCdUDxAj7T2awQ5knPMPawq5Q2RUWM/view?usp=sharing), unzip it and put "FaceReconModel.pb" into ./network subfolder.

6. Run the demo code.

```
python demo.py
```

7. ./input subfolder contains several test images and ./output subfolder stores their reconstruction results. For each input test image, two output files can be obtained after running the demo code:
	- "xxx.mat" : 
		- cropped_img: an RGB image after alignment, which is the input to the R-Net
		- recon_img: an RGBA reconstruction image aligned with the input image (only on Linux).
		- coeff: output coefficients of R-Net.
		- face_shape: vertex positions of 3D face in the world coordinate.
		- face_texture: vertex texture of 3D face, which excludes lighting effect.
		- face_color: vertex color of 3D face, which takes lighting into consideration.
		- lm\_68p: 68 2D facial landmarks derived from the reconstructed 3D face. The landmarks are aligned with cropped_img.
		- lm\_5p: 5 detected landmarks aligned with cropped_img. 
	- "xxx_mesh.obj" : 3D face mesh in the world coordinate (best viewed in MeshLab).

### Training requirements ###

- Training is only supported on Linux. To train new model from scratch, more requirements are needed on top of the requirements listed in the testing stage.
- [Facenet](https://github.com/davidsandberg/facenet) provided by 
Sandberg et al. In our paper, we use a network to exrtact perceptual face features. This network model cannot be publicly released. As an alternative, we recommend using the Facenet from Sandberg et al. This repo uses the version [20170512-110547](https://github.com/davidsandberg/facenet/blob/529c3b0b5fc8da4e0f48d2818906120f2e5687e6/README.md) trained on MS-Celeb-1M. Training process has been tested with this model to ensure similar results.
- [Resnet50-v1](https://github.com/tensorflow/models/blob/master/research/slim/README.md) pre-trained on ImageNet from Tensorflow Slim. We use the version resnet_v1_50_2016_08_28.tar.gz as an initialization of the face reconstruction network.
- [68-facial-landmark detector](https://drive.google.com/file/d/1KYFeTb963jg0F47sTiwqDdhBIvRlUkPa/view?usp=sharing). We use 68 facial landmarks for loss calculation during training. To make the training process reproducible, we provide a lightweight detector that produce comparable results to [the method of Bulat et al.](https://github.com/1adrianb/2D-and-3D-face-alignment). The detector is trained on [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm), [LFW](http://vis-www.cs.umass.edu/lfw/), and [LS3D-W](https://www.adrianbulat.com/face-alignment).

### Training preparation ###

1. Download the [pre-trained weights](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) of Facenet provided by Sandberg et al., unzip it and put all files in ./weights/id_net.
2. Download the [pre-trained weights](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) of Resnet_v1_50 provided by Tensorflow Slim, unzip it and put resnet_v1_50.ckpt in ./weights/resnet.
3. Download the [68 landmark detector](https://drive.google.com/file/d/1KYFeTb963jg0F47sTiwqDdhBIvRlUkPa/view?usp=sharing), put the file in ./network.

### Data pre-processing ###
1. To train our model with custom images，5 facial landmarks of each image are needed in advance for an image pre-alignment process. We recommend using [dlib](http://dlib.net/) or [MTCNN](https://github.com/ipazc/mtcnn). Use these public face detectors to get 5 landmarks, and save all images and corresponding landmarks in <raw_img_path>. Note that an image and its detected landmark file should have same name.
2. Align images and generate 68 landmarks as well as skin masks for training: 

```
# Run following command for data pre-processing. By default, the code uses example images in ./input and saves the processed data in ./processed_data
python preprocess_img.py

# Alternatively, you can set your custom image path and save path
python preprocess_img.py --img_path <raw_img_path> --save_path <save_path_for_processed_data>

```

### Training networks ###
1. Train the reconstruction network with the following command:
```
# By default, the code uses the data in ./processed_data as training data as well as validation data
python train.py

# Alternatively, you can set your custom data path
python train.py --data_path <custom_data_path> --val_data_path <custom_val_data_path> --model_name <custom_model_name>

```
2. Monitoring the training process via tensorboard:
```
tensorboard --logdir=result/<custom_model_name> --port=10001
```
3. Evaluating trained model:
```
python demo.py --use_pb 0 --pretrain_weights <custom_weights>.ckpt
```

## Note

1. An image pre-alignment with 5 facial landmarks is necessary before reconstruction. In our image pre-processing stage, we solve a least square problem between 5 facial landmarks on the image and 5 facial landmarks of the BFM09 average 3D face to cancel out face scales and misalignment. To get 5 facial landmarks, you can choose any open source face detector that returns them, such as [dlib](http://dlib.net/) or [MTCNN](https://github.com/ipazc/mtcnn). However, these traditional 2D detectors may return wrong landmarks under large poses which could influence the alignment result. Therefore, we recommend using [the method of Bulat et al.](https://github.com/1adrianb/2D-and-3D-face-alignment) to get facial landmarks (3D definition) with semantic consistency for large pose images. Note that our model is trained without position augmentation so that a bad alignment may lead to inaccurate reconstruction results. We put some examples in the ./input subfolder for reference.


## Citation

Please cite the following paper if this model helps your research:

	@inproceedings{deng2019accurate,
	    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
	    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
	    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
	    year={2019}
	}
##
The face images on this page are from the public [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset released by MMLab, CUHK.
