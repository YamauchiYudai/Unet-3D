# Swin-Unet
The codes for the work "Medical Image Segmentation using 3D-Unet"


## 1. Prepare data

- The datasets we used are provided by Center for Biomedical Image Computing & Analytics. [Get processed data in this link] (https://www.med.upenn.edu/cbica/brats2020/data.html)). Please go to "Registration/Data Request" page. for details

## 2. Environment

- Please prepare an environment with python=3.10.8, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Train/Test

- We tarin the network using Training dataset and test suing Validation dataset. However Validation dataset's Grand Truth doesn't be uploaded from Center for Biomedical Image Computing & Analytics. If you want to check your result, you have to upload CBICA's Image Processing Portal (ipp.cbica.upenn.edu).


### References
1. Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: learning dense volumetric segmentation from sparse annotation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19 (pp. 424-432). Springer International Publishing.
2. Mehta, Raghav, et al. "QU-BraTS: MICCAI BraTS 2020 challenge on quantifying uncertainty in brain tumor segmentation-analysis of ranking scores and benchmarking results." The journal of machine learning for biomedical imaging 2022 (2022).
