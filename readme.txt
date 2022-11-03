This is reproduce code and improvement on Ren's paper "Neural Blind Deconvolution Using Deep Priors" [1]. 

There is also dataset provided here. 

To run SelfDeblur on lai's dataset [2], simply run:
python selfdeblur_lai.py

To run SelfDeblur on levin's dataset [3], simply run:
python selfdeblur_levin.py

To evaluate the quality of deblurred images in lai's dataset, run:
python evaluation_lai.py

To evaluate the quality of deblurred images in levin's dataset, run:
python evaluation_levin.py


References:
[1] D. Ren, K. Zhang, Q. Wang, Q. Hu and W. Zuo. Neural Blind Deconvolution Using Deep Priors. In IEEE CVPR 2020. 

[2] A. Levin, Y. Weiss, F. Durand, and W. T. Freeman. Understanding and evaluating blind deconvolution algorithms. In IEEE CVPR 2009.

[3] W.-S. Lai, J.-B. Huang, Z. Hu, N. Ahuja, and M.-H. Yang. A comparative study for single image blind deblurring. In IEEE CVPR 2016.