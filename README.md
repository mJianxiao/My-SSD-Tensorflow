# SSD-Tensorflow训练、算法服务器环境配置教程及部分实际运用

>本文将介绍SSD-Tensorflow从环境配置、训练、调参，再到服务器端配置、训练、模型下载及评估的全过程，最后将附上一些实际运用场景

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

添加： `export QT_QPA_PLATFORM='offscreen'`

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

**标注工具：`labelImg`**

**一：下载图片，存入JPEGImages文件夹**

**二：使用labelImg工具给图片打标签**

首先安装: `pip install lxml`

**三：生成训练集、测试集、验证集等**

代码详见：`split_data.py`

***

### 2.2 数据集处理

**一：将数据图片名称转换为6位数字（也可以是其他）**

代码详见：`rename.py`

**二：更改xml中的内容**

更改folder、filename、path

代码详见：`change_filename.py` ，`change_folder.py` ，`change_path.py`

**三：生成.tfrecords文件的代码微调说明**

**修改标签项**——打开datasets文件夹中`pascalvoc_common.py`文件，将自己的标签项填入。

	VOC_LABELS = {
    	'none': (0, 'Background'),
    	'non-motor': (1, 'Non-motor'),
    	'car': (2, 'Motor'),
    	'person': (3, 'People')
	}

**修改读取个数、读取方式**——打开datasets文件夹中的`pascalvoc_to_tfrecords.py`文件

修改67行`SAMPLES_PER_FILES的`个数；

	RANDOM_SEED = 4242
	SAMPLES_PER_FILES = 500
  
修改83行读取方式为`'rb'`；

	filename = directory + DIRECTORY_IMAGES + name + '.jpg'
	image_data = tf.gfile.GFile(filename, 'rb').read()  

如果你的文件不是.jpg格式，也可以修改图片的类型

**生成.tfrecords文件**——打开`tf_convert_data.py`文件，依次点击：`run`、`Edit Configuration`，在`Parameters`中填入以下内容，再运行`tf_convert_data.py`文件，在面板中得到成功信息，可以在tfrecords_文件夹下看到生成的`.tfrecords`文件：

	--dataset_name=pascalvoc
	--dataset_dir=./VOC2007/
	--output_name=voc_2007_train
	--output_dir=./tfrecords_

***

### 2.3 模型训练

**一：修改训练数据shape**——打开datasets文件夹中的`pascalvoc_2007.py`文件

根据自己训练数据修改：`NUM_CLASSES` = 类别数

	TRAIN_STATISTICS = {
    	'none': (0, 0),
    	'non-motor': (682, 4447),
    	'car': (182, 400), # 包含car类型的图片数：182 / 所有图片中包含car框框的数量：400
    	'person': (95, 401),
    	'total': (682, 5248),
	}
	TEST_STATISTICS = {
    	'none': (0, 0),
    	'non-motor': (1, 1),
    	'car': (1, 1),
    	'person': (1, 1),
    	'total': (3, 3),
	}
	SPLITS_TO_SIZES = {
    	'train': 490, # 训练集大小
    	'test': 69, # 测试集大小
	}

计算各类型图片数量和框数量的代码详见：
>calculate.py

**二：修改类别个数**——打开nets文件夹中的`ssd_vgg_300.py`或者`ssd_vgg_512.py`文件

根据自己训练类别数修改96和97行：**等于类别数+1** （SSD512对应78和79行）

	num_classes=4,
	no_annotation_label=4,

打开`eval_ssd_network.py`文件

修改66行的类别个数：**等于类别数+1**

	tf.app.flags.DEFINE_integer(
	    'num_classes', 4, 'Number of classes to use in the dataset.')

**三：修改训练步数epoch**——打开`train_ssd_network.py`文件

27行的数据格式，用CPU训练则改为`'NHWC'`，用GPU训练则为`'NCHW'`

修改135行的类别个数：**等于类别数+1**

	tf.app.flags.DEFINE_integer(
    	'num_classes', 4, 'Number of classes to use in the dataset.')

修改154行训练总步数，None会无限训练下去（这里设置的6万步）

	tf.app.flags.DEFINE_integer('max_number_of_steps', 60000,
                            	'The maximum number of training steps.')

关于模型运行保存的参数：

	tf.app.flags.DEFINE_integer(
    	'save_interval_secs', 600, # 最好设置为600
    	'The frequency with which the model is saved, in seconds.')

**四：开始训练模型**

可加载vgg_16模型开始训练，也可以重头开始训练模型

打开`train_ssd_network.py`文件，依次点击：`run`、`Edit Configuration`，在`Parameters`中填入以下内容，再运行`train_ssd_network.py`文件

	--train_dir=./train_model/
	--dataset_dir=./tfrecords_/
	--dataset_name=pascalvoc_2007
	--dataset_split_name=train
	--model_name=ssd_300_vgg
	--checkpoint_path=./checkpoints/vgg_16.ckpt # 加载预训练模型
	--checkpoint_model_scope=vgg_16
	--checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box
	--trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box
	--save_summaries_secs=600
	--save_interval_secs=100
	--weight_decay=0.0005
	--optimizer=adam
	--learning_rate=0.001
	--learning_rate_decay_factor=0.94
	--batch_size=4 # 显卡越好可设置越大
	--gpu_memory_fraction=0.9

训练SSD512参数如下：

	--train_dir=./train_model/ 
	--dataset_dir=./tfrecords_/ 
	--dataset_name=pascalvoc_2007 
	--dataset_split_name=train 
	--model_name=ssd_512_vgg 
	--checkpoint_path=./checkpoints/vgg_16.ckpt 
	--checkpoint_model_scope=vgg_16 
	--checkpoint_exclude_scopes=ssd_512_vgg/block7,ssd_512_vgg/block7_box,ssd_512_vgg/block8,ssd_512_vgg/block8_box,ssd_512_vgg/block9,ssd_512_vgg/block9_box,ssd_512_vgg/block10,ssd_512_vgg/block10_box,ssd_512_vgg/block11,ssd_512_vgg/block11_box,ssd_512_vgg/block12,ssd_512_vgg/block12_box 
	--save_summaries_secs=600 
	--save_interval_secs=600 
	--weight_decay=0.0005 
	--optimizer=adam 
	--learning_rate=0.001 
	--learning_rate_decay_factor=0.94 
	--batch_size=32 
	--gpu_memory_fraction=0.9

重头开始训练参数如下（不加载预训练模型）：

	--train_dir=./train_model/ 
	--dataset_dir=./tfrecords_/ 
	--dataset_name=pascalvoc_2007 
	--dataset_split_name=train 
	--model_name=ssd_512_vgg 
	--save_summaries_secs=600 
	--save_interval_secs=600 
	--weight_decay=0.0005 
	--optimizer=adam 
	--learning_rate=0.001 
	--learning_rate_decay_factor=0.94 
	--batch_size=32 
	--gpu_memory_fraction=0.9

部分参数说明：

`--save_interval_secs`是训练多少次保存参数的步长；  
`--optimizer`是优化器；  
`--learning_rate`是学习率；  
`--learning_rate_decay_factor`是学习率衰减因子。

训练结束可以在`train_model`文件夹下看到生成的参数文件

***

**五：测试验证**

生成测试集的` .tfrecords`文件。将测试图片转换为tfrecords

打开`eval_ssd_network.py`文件，依次点击：`run`、`Edit Configuration`，在`Parameters`中填入以下内容，再运行`eval_ssd_network.py`文件

	--eval_dir=./ssd_eval_log/ 
	--dataset_dir=./tfrecords_/tfrecord/ 
	--dataset_name=pascalvoc_2007 
	--dataset_split_name=test 
	--model_name=ssd_512_vgg 
	--checkpoint_path=./train_model/model.ckpt-60000 
	--batch_size=1

## 3 结果展示

1、在日志中，选取最后一次生成模型作为测试模型进行测试；

2、在demo文件夹下放入测试图片；

3、最后运行在notebooks文件夹下的demo_test.py测试文件。

## 4 SSD模型在实际项目中的一些运用

### 智慧城市街面管理

***

**判定告警区域**（载入黑白图）

    # 判定所属区域，载入区域图

    segmentation_path = './segmentation/'

    # 区域1
    area1 = 'area001.jpg'
    img1 = cv2.imread(segmentation_path + aera1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _, img1 = cv2.threshold(img1, 175, 255, cv2.THRESH_BINARY_INV)
    aera1_list = np.argwhere(img1 > 250).tolist()

    # 区域2
    area2 = 'area002.jpg'
    img2 = cv2.imread(segmentation_path + area2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, img1 = cv2.threshold(img2, 175, 255, cv2.THRESH_BINARY_INV)
    area2_list = np.argwhere(img2 > 250).tolist()

    # 区域3
    area3 = 'area003.jpg'
    img3 = cv2.imread(segmentation_path + area3)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    _, img1 = cv2.threshold(img3, 175, 255, cv2.THRESH_BINARY_INV)
    area3_list = np.argwhere(img3 > 250).tolist()

***

**剔除不属于任何区域的识别物，判定所属区域**

    new_cls = []
    new_score = []
    new_box = []
    area = []

    for i in range(len(rclasses)):
        a = (rbboxes[i][1] + rbboxes[i][3]) / 2
        b = rbboxes[i][2]
        coord = [int(b * height) - 1, int(a * width) - 1]
        if coord in area1_list:
            new_cls.append(1)
            new_score.append(rscores[i])
            new_box.append(rbboxes[i])
            area.append('area001')
        elif coord in area2_list:
            new_cls.append(1)
            new_score.append(rscores[i])
            new_box.append(rbboxes[i])
            area.append('area002')
        elif coord in area3_list:
            new_cls.append(1)
            new_score.append(rscores[i])
            new_box.append(rbboxes[i])
            area.append('area003')
        else:
            pass
    new_cls = np.array(new_cls)
    new_score = np.array(new_score)
    new_box = np.array(new_box)

***

**根据告警区域输出电子围栏**

    origin = cv2.imread('./pic/' + camera + '-' + str(now_time) + '.png')
    quyu = cv2.imread('./segmentation/area001.jpg')
    quyu = cv2.resize(quyu, (origin.shape[1], origin.shape[0]))
    _, th = cv2.threshold(cv2.cvtColor(quyu.copy(), cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    th = 255 - th
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    final_img = cv2.drawContours(origin.copy(), contours, -1, (0, 0, 255), 5)
    b, g, r = cv2.split(final_img)
    final_img = cv2.merge([r, g, b])
    plt.figure(figsize=(19.2, 10.8))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(final_img)
    plt.savefig('./pic/' + camera + '-' + str(now_time) + '.png')

>PNG转JPG减小输出告警图片大小

    im = Image.open('./pic/' + camera + '-' + str(now_time) + '.png')
    im = im.convert('RGB')
    im.save('./pic/' + camera + '-' + str(now_time) + '.jpg')
    os.remove('./pic/' + camera + '-' + str(now_time) + '.png')

***

**防止误报去除一些重合框部分**

以垃圾厢房为例：

    # 垃圾厢房去除人车重合部分

    remove_box = []
    for i in range(len(new_cls)):
        if new_cls[i] == 2 or new_cls[i] == 3 or new_cls[i] == 4 or new_cls[i] == 5:
            remove_box.append(new_box[i])
        else:
            pass

    lajmy = []
    lajmy_score = []
    lajmy_box = []
    if remove_box == []:
        for i in range(len(new_cls)):
            if new_cls[i] == 1:
                lajmy_box.append(new_box[i])
                lajmy.append(new_cls[i])
                lajmy_score.append(new_score[i])
    else:
        for i in range(len(new_cls)):
            if new_cls[i] == 1:
                t = []
                for j in range(len(remove_box)):
                    t.append(IOS(new_box[i], remove_box[j]))
                if max(t) < 0.2:
                    lajmy_box.append(new_box[i])
                    lajmy.append(new_cls[i])
                    lajmy_score.append(new_score[i])
                else:
                    pass
            else:
                pass

    lajmy = np.array(lajmy)
    lajmy_box = np.array(lajmy_box)
    lajmy_score = np.array(lajmy_score)

>计算两个框重叠度的相关函数定义：

	def IOU(rectA, rectB):
    	W = min(rectA[2], rectB[2]) - max(rectA[0], rectB[0])
    	H = min(rectA[3], rectB[3]) - max(rectA[1], rectB[1])
    	if W <= 0 or H <= 0:
        	return 0;
    	SA = (rectA[2] - rectA[0]) * (rectA[3] - rectA[1])
    	SB = (rectB[2] - rectB[0]) * (rectB[3] - rectB[1])
    	cross = W * H
    	return cross / (SA + SB - cross)


	def IOS(rectA, rectB):
    	W = min(rectA[2], rectB[2]) - max(rectA[0], rectB[0])
    	H = min(rectA[3], rectB[3]) - max(rectA[1], rectB[1])
    	if W <= 0 or H <= 0:
        	return 0;
    	SA = (rectA[2] - rectA[0]) * (rectA[3] - rectA[1])
    	SB = (rectB[2] - rectB[0]) * (rectB[3] - rectB[1])
    	min_S=min(SA,SB)
    	cross = W * H
    	return cross/min_S

***

**告警推送定义：当识别出异常数满足一定数量后才认定为告警**

	# 检测到目标物后开始计时，当出现时间超过count进行告警，同时告警后如果识别结果未改变，不重复告警
	
	cls_list = rclasses.tolist()
	cls_list.sort()
	if cls_list != [] and cls_list != temp:
		count = count + 1
		if count > 50:
			plt_bboxes(frame_chu, rclasses, rscores, area, rbboxes, now_time)
			kafka_prod(rclasses, camera, now_time, area)
			count = 0
			temp = rclasses.tolist()
			temp.sort()
		if temp != [] and cls_list == 0:
			reset = reset + 1
			if reset > 6000:
				temp = []
				reset = 0

***

**告警推送方式**（Kafka）

	def kafka_prod(collect_class, camera, now_time, area):
    	try:
        	 = KafkaProducer(bootstrap_servers=server)
        	for i in range(len(collect_class)):
            	camera_d = camera
            	pic_name = camera + '-' + str(now_time)  + '.jpg'
            	data = json.dumps(
                	{
                    	'data': {
                        	'cameraId': camera_d,
                       		'disorderId':area[i],
                        	'createTime': str(now_time),
                        	'type': str(collect_class[i]),
                        	'imgUrl': pic_name
                    	},
                    	'type': 'dustbin'
                	})
            	print('data', data)
            	producer.send(topic, data.encode('utf-8'))
            	producer.close()
    	except:
        	pass
    	return None

***

**针对服务器性能不够（无法同时处理上百路视频流）可采用视频流轮循方式读取视频识别**

    num = n		# 识别推送过来所有视频流的第n个
    temp = []
    count = 0

    while True:
        with open('./add.csv', 'r') as f:
            reader = csv.reader(f)
            add  = list(reader)
        camera = add[num][0]
        url = add[num][1]

根据第n个视频流对应的`camera id`载入相应区域图

    Area1 = camera[-4:] + '.jpg'
    img1 = cv2.imread(segmentation_path + Area1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _, img1 = cv2.threshold(img1, 175, 255, cv2.THRESH_BINARY_INV)
    Area1_list = np.argwhere(img1 > 250).tolist()

>接收推送过来的视频流保存方式：

	from kafka import KafkaConsumer
	import json
	import csv
	import time


	consumer = KafkaConsumer('videoUrl', bootstrap_servers="10.100.1.5:9092", auto_offset_reset='latest')
	for msg in consumer:
    	value = msg.value.decode(encoding="utf-8")
    	id_add = json.loads(value)
    	f = open('add.csv', 'w', encoding='utf-8', newline='')
    	csv_writer = csv.writer(f)
    	for i in range(len(id_add)):
        	csv_writer.writerow([id_add[i]['cameraMem'], id_add[i]['url']])
    	f.close

存储为csv后可实时更新

***