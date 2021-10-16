# coco数据集的使用

## 1. 标注类型 (.json)

包括三种类型：

**object instances（目标实例）**,**object keypoints（目标上的关键点）**, and **image captions（看图说话) **

每种类型又包含了**训练(train.json)**和**验证(val.json)**，他们是 `.json`  文件

## 2.基本json结构体

这3种类型共享下面所列的基本类型，包括**info、image、license **这三种类型是所有json中都共享的，只有image比较重要，而**annotation**类型则呈现出了多态:

``` php
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
}
    
info{
    "year": int,
    "version": str,
    "description": str,
    "contributor": str,
    "url": str,
    "date_created": datetime,
}
license{
    "id": int,
    "name": str,
    "url": str,
} 
image{
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
    "license": int,
    "flickr_url": str,
    "coco_url": str,
    "date_captured": datetime,
}
```

### 2.1 info类型

以下是一个实例

``` json
"info":{
	"description":"This is stable 1.0 version of the 2014 MS COCO dataset.",
	"url":"http:\/\/mscoco.org",
	"version":"1.0",
    "year":2014,
	"contributor":"Microsoft COCO group",
	"date_created":"2015-01-27 09:11:52.357475"
},
```

不重要，可以忽略。

### 2.3 images类型

Images是包含多个image实例的数组，对于一个image类型的实例，每一个image指向一张图片：

license、coco_url、data_captured和flickr_url这几个key指向的信息大概了解下就行

**file_name**，指向的是一个字符串，是jpg的文件名；

**id**指向的数字是每张图片特有的一个标志，数字不重复，可以看作是图片的身份信息，就像身份证那一串数字一样。

```json
{
	"license":3,  
	"file_name":"COCO_val2014_000000391895.jpg",
	"coco_url":"http:\/\/mscoco.org\/images\/391895",
	"height":360,"width":640,"date_captured":"2013-11-14 11:18:45",
	"flickr_url":"http:\/\/farm9.staticflickr.com\/8186\/8119368305_4e622c8349_z.jpg",
	"id":391895
},
```

### 2.3 licenses 类型

**License这个key指向的信息也可以忽略不计**

licenses是包含多个license实例的数组，对于一个license类型的实例：

``` json
{
	"url":"http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/",
	"id":1,
	"name":"Attribution-NonCommercial-ShareAlike License"
},
```

## 3. object instances 类型的标注格式

### 3.1 整体文件格式

上面说到，这个包括`instances_train` 和`instance_val`格式

每个文件中为五段信息的重复，内容如下：

``` json
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}
```

是的，你打开这两个文件，虽然内容很多，但从文件开始到结尾按照顺序就是这5段。其中，info、licenses、images这三个结构体/类型 在上一节中已经说了，在不同的JSON文件中这三个类型是一样的，定义是共享的。不共享的是**annotation**和**category**这两种结构体，他们在不同类型的JSON文件中是不一样的。

**images**数组、**annotations**数组、**categories**数组的元素数量是相等的，等于图片的数量。

### 3.2 annotation字段

annotations字段是包含多个annotation实例的一个数组，annotation类型本身又包含了一系列的字段，如这个目标的category id和segmentation mask。

```json
annotation{
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}
```

**segmentation**格式取决于这个实例是一个单个的对象（即iscrowd=0，将使用polygons格式）还是一组对象（即iscrowd=1，将使用RLE格式）。如下所示：

*注意，单个的对象（iscrowd=0)可能需要多个**polygon**来表示，比如这个对象在图像中被挡住了。而iscrowd=1时（将标注一组对象，比如一群人）的segmentation使用的就是**RLE**格式。

segmentation指向是的一个套着两个list的东西，里面一堆数字的含义是像素级分割得到的物体边缘坐标；

**area**指向该segmentation的面积

**iscrowd**目前来看都指向0，表示没有重叠吧，有重叠指向1

**category_id**字段存储的是当前对象所属的category的id，所属的supercategory的name。

**image_id**就是前面images中存储的id 

**bbox**指向的就是物体的框；

**id**不同于images中的id，images中的id是每幅图片的身份编号，而此处的id是每个框的身份编号，注意区分。

#### 实例

``` json
{
	"segmentation": [[510.66,423.01,511.72,420.03,510.45,416.0,510.34,413.02,510.77,410.26,\
			510.77,407.5,510.34,405.16,511.51,402.83,511.41,400.49,510.24,398.16,509.39,\
			397.31,504.61,399.22,502.17,399.64,500.89,401.66,500.47,402.08,499.09,401.87,\
			495.79,401.98,490.59,401.77,488.79,401.77,485.39,398.58,483.9,397.31,481.56,\
			396.35,478.48,395.93,476.68,396.03,475.4,396.77,473.92,398.79,473.28,399.96,\
			473.49,401.87,474.56,403.47,473.07,405.59,473.39,407.71,476.68,409.41,479.23,\
			409.73,481.56,410.69,480.4,411.85,481.35,414.93,479.86,418.65,477.32,420.03,\
			476.04,422.58,479.02,422.58,480.29,423.01,483.79,419.93,486.66,416.21,490.06,\
			415.57,492.18,416.85,491.65,420.24,492.82,422.9,493.56,424.39,496.43,424.6,\
			498.02,423.01,498.13,421.31,497.07,420.03,497.07,415.15,496.33,414.51,501.1,\
			411.96,502.06,411.32,503.02,415.04,503.33,418.12,501.1,420.24,498.98,421.63,\
			500.47,424.39,505.03,423.32,506.2,421.31,507.69,419.5,506.31,423.32,510.03,\
			423.01,510.45,423.01]],
	"area": 702.1057499999998,
	"iscrowd": 0,
	"image_id": 289343,
	"bbox": [473.07,395.93,38.65,28.67],
	"category_id": 18,
	"id": 1768
},
```

### 3.3 category

categories是一个包含多个category实例的数组，而category结构体描述如下：

```json
{
    "id": int,
    "name": str,
    "supercategory": str,
}
```

以下是一个实例

``` json
{
	"supercategory": "person",
	"id": 1,
	"name": "person"
},
{
	"supercategory": "vehicle",
	"id": 2,
	"name": "bicycle"
},
```

## 4. Object Keypoint 类型的标注格式

### 4.1 整体文件格式

Object Keypoint这种格式的文件从头至尾按照顺序分为以下段落，看起来和Object Instance一样啊：

```json
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}
```

不共享的是annotation和category这两种结构体，他们在不同类型的JSON文件中是不一样的。

images数组和annotations数组的元素数量是相等的，等于图片的数量。

### 4.2 annotations字段

这个类型中的annotation结构体包含了Object Instance中annotation结构体的所有字段，再加上2个额外的字段。

新增的keypoints是一个长度为3*k的数组，其中k是category中keypoints的总数量。每一个keypoint是一个长度为3的数组，第一和第二个元素分别是x和y坐标值，第三个元素是个标志位v，v为0时表示这个关键点没有标注（这种情况下x=y=v=0），v为1时表示这个关键点标注了但是不可见（被遮挡了），v为2时表示这个关键点标注了同时也可见。

num_keypoints表示这个目标上被标注的关键点的数量（v>0），比较小的目标上可能就无法标注关键点。

```php
annotation{
    "keypoints": [x1,y1,v1,...],
    "num_keypoints": int,
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}
```

例如：

``` json
{
	"segmentation": [[125.12,539.69,140.94,522.43,100.67,496.54,84.85,469.21,73.35,450.52,104.99,342.65,168.27,290.88,179.78,288,189.84,286.56,191.28,260.67,202.79,240.54,221.48,237.66,248.81,243.42,257.44,256.36,253.12,262.11,253.12,275.06,299.15,233.35,329.35,207.46,355.24,206.02,363.87,206.02,365.3,210.34,373.93,221.84,363.87,226.16,363.87,237.66,350.92,237.66,332.22,234.79,314.97,249.17,271.82,313.89,253.12,326.83,227.24,352.72,214.29,357.03,212.85,372.85,208.54,395.87,228.67,414.56,245.93,421.75,266.07,424.63,276.13,437.57,266.07,450.52,284.76,464.9,286.2,479.28,291.96,489.35,310.65,512.36,284.76,549.75,244.49,522.43,215.73,546.88,199.91,558.38,204.22,565.57,189.84,568.45,184.09,575.64,172.58,578.52,145.26,567.01,117.93,551.19,133.75,532.49]],
	"num_keypoints": 10,
	"area": 47803.27955,
	"iscrowd": 0,
	"keypoints": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,309,1,177,320,2,191,398,2,237,317,2,233,426,2,306,233,2,92,452,2,123,468,2,0,0,0,251,469,2,0,0,0,162,551,2],
	"image_id": 425226,"bbox": [73.35,206.02,300.58,372.5],"category_id": 1,
	"id": 183126
},
```

### 4.3 category

对于每一个category结构体，相比Object Instance中的category新增了2个额外的字段，

**keypoints**是一个长度为k的数组，包含了每个关键点的名字；

**skeleton**定义了各个关键点之间的连接性（比如人的左手腕和左肘就是连接的，但是左手腕和右手腕就不是）。目前，COCO的keypoints只标注了person category （分类为人）。

``` php
{
    "id": int,
    "name": str,
    "supercategory": str,
    "keypoints": [str],
    "skeleton": [edge]
}

```

某个实例：

```json
{
	"supercategory": "person",
	"id": 1,
	"name": "person",
	"keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
	"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
}
```



## 5.Image Caption的标注格式

### 5.1整体JSON文件格式

**Image Caption**这种格式的文件从头至尾按照顺序分为以下段落，看起来和Object Instance一样，不过没有最后的categories字段：

```php
{
    "info": info,
    "lisupercategorycenses": [license],
    "images": [image],
    "annotations": [annotation]
}
```

不共享的是annotations这种结构体，它在不同类型的JSON文件中是不一样的。所以此处的annotation和上面的又不一样

***  images数组的元素数量等于图片的数量。 

### 5.2 annotation语句

这个类型中的annotation用来存储描述图片的语句。每个语句描述了对应图片的内容，而每个图片至少有5个描述语句（有的图片更多）。annotation定义如下：

```json
annotation{
    "id": int,
    "image_id": int,
    "caption": str
}
```

以下是一个描述实例

```json
{
	"image_id": 179765,
	"id": 38,
    "caption": "A black Honda motorcycle parked in front of a garage."
},
```

