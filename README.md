# SSD-Tensorflow训练、算法服务器环境配置及部署教程

>本文将介绍SSD-Tensorflow从环境配置、训练、调参，再到服务器端配置、训练、模型下载及评估的全过程

## 1 环境配置

### 1.0 显卡驱动安装

#### CentOS 安装步骤

**查看内核版本（需安装对应版本的开发包）**

`uname -r`

**查看可安装的开发包**

`yum list | grep kernel-devel`

**安装依赖**

`yum install gcc dkms kernel-devel kernel-doc kernel-headers`

**查看nouveau有没有被禁用**

`lsmod | grep nouveau`

**阻止 nouveau 模块的加载:**

修改/etc/modprobe.d/blacklist.conf 文件：

	blacklist nouveau  
	options nouveau modeset=0

如果不存在，执行：

`echo -e "blacklist nouveau\noptions nouveau modeset=0" > /etc/modprobe.d/blacklist.conf`

**重新建立initramfs image文件**


`mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak`  
`dracut /boot/initramfs-$(uname -r).img $(uname -r)`

重启：`reboot`

**执行安装脚本**

赋权：`chmod u+x NVIDIA-Linux-x86_64-415.13.run`

内核改为自己系统对应版本：

`./NVIDIA-Linux-x86_64-375.39.run --kernel-source-path=/usr/src/kernels/3.10.0-862.el7.x86_64`

***

### 1.1 Anaconda3安装及搭建虚拟训练环境

**安装前（防止报错）：**
`yum -y install bzip2`

**安装：**
`bash anaconda3.5.2.0-Linux-x86_64.sh`

**添加环境变量（可安装过程中自动添加）：**
`vi ~/.bashrc`

添加 
> export PATH="/root/anaconda3/bin:$PATH"

`source ~/.bashrc`

**创建虚拟环境指令：**
`conda create -n your_env_name python=3.7`

**激活虚拟环境指令：**
`conda activate your_env_name`

**关闭虚拟环境:**
`conda deactivate`

**删除指定虚拟环境（如果为空环境则此命令失效，也就是未指定Python版本的时候）：**
`conda remove -n my_Anaconda_env --all`

**删除指定虚拟环境（无论此环境是否为空）：**
`conda env remove -n my_Anaconda_env`

**安装Anaconda收录的Python包:**
`conda install package`

**安装所有Python收录的包:**
`pip install package`
***
**解决pip安装时速度慢及time-out的问题**

修改 `~/.pip/pip.conf` 没有则创建一个


	[global] 
	timeout = 6000    
	index-url = https://pypi.tuna.tsinghua.edu.cn/simple  
	[install]  
	trusted-host=mirrors.aliyun.com

* * *

#### 安装 Tensorflow-GPU

`pip install tensorflow-gpu==1.14.0`

**报错及解决方案**

> ERROR: Cannot uninstall 'Werkzeug'.  

`pip install --ignore-installed six tensorflow-gpu==1.14.0`

> setuptools版本过低

`pip install --upgrade --ignore-installed setuptools`

> ERROR: Cannot uninstall 'wrapt'.

`pip install -U --ignore-installed wrapt enum34 simplejson netaddr`

> QXcbConnection: Could not connect to display

`vim ~/.bashrc`

`export QT_QPA_PLATFORM='offscreen'`

`source ~/.bashrc`

#### 安装 OpenCV

`pip install opencv-python==4.1.1.26`

***

### 1.2 CUDA安装

**赋权：**
`chmod a+x cuda_10.0.130_410.48_linux.run`

**安装：**
`sudo sh cuda_10.0.130_410.48_linux.run`

**添加环境变量：**
`vi ~/.bashrc`

添加

	export PATH=/usr/local/cuda-10.0/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
	export CUDA_HOME=/usr/local/cuda

更新：`source ~/.bashrc`

验证安装：`nvcc -V`

**编译samples例子**

*编译并测试设备 deviceQuery：*

	cd /usr/local/cuda-10.0/samples/1_Utilities/deviceQuery
	sudo make
	./deviceQuery

*编译并测试带宽 bandwidthTest：*

	cd ../bandwidthTest
	sudo make
	./bandwidthTest

**报错及解决方案**

	Missing recommended library: libGLU.so
	Missing recommended library: libX11.so
	Missing recommended library: libXi.so
	Missing recommended library: libXmu.so

安装依赖库文件：

	yum install libGLU*
	yum install libX11*
	yum install libXi*
	yum install libXmu*

安装时报错：`toolkit installation failed using unsupported compiler`

`sudo sh cuda_10.0.130_410.48_linux.run -override`


### 1.3 cuDNN安装

**解压:**
`tar xvf cudnn-10.0-linux-x64-v7.6.3.30`

**复制到cuda目录**

	sudo cp cuda/include/* /usr/local/cuda/include
	sudo cp cuda/lib64/*   /usr/local/cuda/lib64

## 2 SSD模型训练

### 2.1 图片标注

### 2.2 数据集处理