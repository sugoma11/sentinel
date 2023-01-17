# Computer vision in remote Earth sensing
---
This is a part of my bachelor degree graduation project. Its in work now.
Now project consists of three parts:

1. Feature extraction:

It's a [main.py in /pic_features](https://github.com/sugoma11/sentinel/blob/master/pic_features/main.py). There are extracting hystograms in RGB-HSV, finding statistical estimators (mean, dispresion, skew etc) for ROI, displaying mean for ROI in RGB image in this script. Also a thought analyse diff between ROI and clear out-port zone. It's a very naive and simple approach, but this idea became to me when I just began to learn CV, so I had to finish it. In this [notebook](https://github.com/sugoma11/itmo_grad_project/blob/main/project.ipynb) more info about research. And this problem should be solved by another ways e.g with deep learning.

Result of work:

![alt text](https://raw.githubusercontent.com/sugoma11/sentinel/master/pic_features/featured/gelendz_tst/2021-01-20-00_00_2021-01-20-23_59_Sentinel-2_L2A_True_color.png.jpg)

2. Data visualization:

[3-dim.py in /data_visualization/3-dim](https://github.com/sugoma11/sentinel/blob/master/data_visualization/3-dim/3dim_graphing.py). It will plot data in RBG-space, also it can plot hyperplane and 2-dim graphics. 
You can see below why using means from RGB channels is a bad idea (we can't divide data):
![alt text](https://raw.githubusercontent.com/sugoma11/sentinel/master/data_visualization/3-dim/estimator_mean_RGB_space.png)

Hyperplane for two classes:
![alt text](https://raw.githubusercontent.com/sugoma11/sentinel/master/data_visualization/3-dim/hyperpane_dirty_vs_clear.png)


3. Testing models:

[test_models.py in /test_models](https://github.com/sugoma11/sentinel/blob/master/test_models/test_models.py) is a script that visualize predictions of model right on source image. Green means clear water, blue means ship, red means pollution. It's possible to mark any zone on image. Not the best model working in gifs below:

<center>
<img src="https://media.giphy.com/media/idQGn2CMa4JIOBNRAT/giphy.gif" width="700" height="400" align="middle"/>

&nbsp;

<img src="https://media.giphy.com/media/PkXwpwALOkxvOIcjeL/giphy.gif" width="700" height="400" align="middle"/>
</center>
