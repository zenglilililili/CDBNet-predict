# 一、文件介绍
## predictOne.py
运行此文件可预测单个病例。\
输入的是data/ct中的.npy数据，输出保存在data/predict中，输入与输出id一一对应\

## bestModel文件夹
保存需要用到的已经训练完成的一个分类模型model1.pkl和一个分割模型model2.pkl\

## data文件夹
data/ct/XX.npy表示单个病人的一张切片，是处理后的CT图像数据。由dicom读出后，转换成灰度值并设置截断区间为[-128,256]，然后归一化所得。大小为256×256的二位矩阵\
data/label/XX.npy是医生勾画的标准CTV的256*256的二位矩阵，其中值为1的像素点表示该处为目标区域，值为0表示不为目标区域\
data/predict/XX.jpg是模型预测的每张切片CTV图像，与data/label/XX.npy中对应id图像越相似表示分割效果越好\

## models文件夹
保存的是训练过程中用到的模型的构建代码\

# 二、代码运行
## 运行环境
python 3.6


Package              Version\
-------------------- -----------\
absl-py              0.10.0\
astor                0.8.1\
beautifulsoup4       4.9.3\
certifi              2020.6.20\
cycler               0.10.0\
decorator            4.1.2\
future               0.18.2\
gast                 0.4.0\
google-pasta         0.2.0\
graphviz             0.16\
grpcio               1.32.0\
h5py                 2.10.0\
hausdorff            0.2.6\
imageio              2.10.1\
importlib-metadata   2.0.0\
joblib               0.17.0\
Keras-Applications   1.0.8\
Keras-Preprocessing  1.1.2\
kiwisolver           1.2.0\
llvmlite             0.36.0\
lxml                 4.6.1\
Markdown             3.3\
matplotlib           2.0.2\
networkx             1.11\
nibabel              3.1.1\
numba                0.53.1\
numpy                1.19.5\
olefile              0.44\
opencv-python        3.4.3.18\
packaging            20.4\
pandas               1.1.3\
Pillow               8.4.0\
pip                  20.2.3\
protobuf             3.13.0\
pydicom              2.0.0\
pyparsing            2.4.7\
python-dateutil      2.8.1\
pytz                 2020.1\
PyWavelets           0.5.2\
scikit-image         0.13.0\
scikit-learn         0.23.2\
scipy                1.5.4\
setuptools           50.3.0\
SimpleITK            2.0.2\
six                  1.15.0\
sklearn              0.0\
soupsieve            2.0.1\
tensorboard          1.15.0\
tensorboardX         2.1\
tensorflow-estimator 1.14.0\
tensorflow-gpu       1.14.0\
termcolor            1.1.0\
threadpoolctl        2.1.0\
**torch                1.6.0+cu101**\
torchvision          0.7.0+cu101\
tqdm                 4.51.0\
Werkzeug             1.0.1\
wheel                0.29.0\
wrapt                1.12.1\
zipp                 3.3.0\

## 运行命令
*激活环境*
以下命令为ubuntu终端命令,pytorch_env是我的环境名称
```
source activate pytorch_env
```
*运行预测代码*
```
python predictOne.py
```
