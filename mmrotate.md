## mmrotate环境安装——>[官网](https://mmrotate.readthedocs.io/en/latest/install.html)

###### 1. 新建conda环境

- conda create -n FAIR python=3.7 -y

- conda activate FAIR

- 安装pytorch， torchvision， cudatoolkit版本

   参考Pytroch官网 ==保证cudatoolkit兼容物理机cuda==

  > 示例：conda install pytorch\==1.10.1  torchvision\==0.11.2 torchaudio\==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

###### 2.安装mmrotate

> pip install openmim
>
> mim install mmcv-full
>
> mim install mmdet
>
> git clone https://github.com/open-mmlab/mmrotate.git
>
> cd mmrotate
>
> pip install -r requirements/build.txt
>
> pip install -v -e .

###### 3. （==备选==）安装swig， dota相关可能会用到

> sudo apt-get install swig
>
> cd DOTA_devkit
> swig -c++ -python polyiou.i
> python setup.py build_ext --inplace

###### 4. DOTA数据集 SOTA排名

https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1

###### 5. ==基于mmrotate训练==

参考网址：[https://blog.csdn.net/qq_43581224/article/details/123838415](https://blog.csdn.net/qq_43581224/article/details/123838415)



###### 6. 测试和训练命令（mmrotate目录下）

>  # train
> python 		tools/train.py 		/root/mmrotate/configs/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc.py 
> python tools/train.py /root/OrientedRepPoints/configs/dota/orientedrepoints_r50_demo.py
>
> 解释：python    tools/train.py    配置文件   --work_dir     保存当前实验的模型检查点和日志的目录(默认创建work_dir文件夹)
>
>  # test
> python		 tools/test.py 		/root/mmrotate/configs/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc.py /root/mmrotate/work_dirs/r3det_kfiou_ln_r50_fpn_1x_dota_oc/epoch_12.pth 		--out out.pkl
>
> 解释：python    tools/test.py    配置文件    checkpoint文件      --out   输出路径
