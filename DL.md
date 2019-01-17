吴恩达 
https://www.bilibili.com/video/av35144344?spm_id_from=333.788.b_765f64657363.1

无痛的机器学习第一季

https://zhuanlan.zhihu.com/p/22464594


深度学习经典、前沿论文讲解（知乎专栏）
https://zhuanlan.zhihu.com/liuyan0612

https://blog.csdn.net/v_JULY_v/article/details/80170182

小豆豆之人脸检测（知乎）
https://zhuanlan.zhihu.com/p/50929457

https://zhuanlan.zhihu.com/c_181106394


目标检测入门：
https://juejin.im/entry/5a98edc06fb9a028c9797fc9

检测算法总结：
https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html

https://zhuanlan.zhihu.com/p/39579528

 1	Faster RCNN
https://zhuanlan.zhihu.com/p/30621997


2	Yolo
https://blog.csdn.net/hrsstudy/article/details/70305791

源码解析：
https://blog.csdn.net/column/details/13752.html


3	SSD
https://blog.csdn.net/u010167269/article/details/52563573   （写的很好）

https://zhuanlan.zhihu.com/p/29410169 （写的很好）

https://zhuanlan.zhihu.com/p/31427288

https://zhuanlan.zhihu.com/p/24954433

5	RFCN
https://zhuanlan.zhihu.com/p/47579399


http://caffe.berkeleyvision.org/tutorial/solver.html

https://ethereon.github.io/netscope/#/editor


目标检测算法总结与对比【写的好】
https://www.cnblogs.com/venus024/p/5590044.html


https://lgwangh.github.io/


https://lgwangh.github.io/


https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html

超分（https://github.com/huangzehao/caffe-vdsr）

tensorflow(https://blog.csdn.net/xierhacker/article/category/6511974)

pytorch(https://blog.csdn.net/u014380165/article/details/78525273)

降噪的也有一系列代码可以参考（https://github.com/wenbihan/reproducible-image-denoising-state-of-the-art）

https://zhuanlan.zhihu.com/p/32206896

专利查询网站
https://patentscope.wipo.int/search/zh/result.jsf

caffe未定义的层：
CReLU 层为PVANet中特殊的层结构，其结构如下，在Caffe中并没有标准的CReLU
层作为单独的一层。
注意：在CReLU 最早提出的论文中《Understanding and Improving Convolutional Neural
Networks via Concatenated Rectified Linear Units》（http://cn.arxiv.org/abs/1603.05201），
并没有如图3-8 所述的Scale/Shift层，也即没有训练参数。不过，在NNIE mapper支
持的单独CReLU 层的实现中，是以PVANet网络中的方式来实现，Caffemodel模型文
件中需要包含Scale层对应的参数。如果要使用不同与此方式实现的CReLU 层，可以
通过多个Caffe标准层组合实现。

PassThrough层为Yolo v2中的一个自定义层

Depthwise Convolution 层为Xception 网络中的自定义层，在caffe 框架中同样也是没有
进行标准定义的。Depthwise Convolution 实现的操作为针对输入的每一个channel，单
独做K*K的卷积，假设输入的channel是M1，输出结果的channel依然是M1。因
此，可以认为有M1个K*K*1 的卷积kernel。


RReLU 层即Randomized Leaky ReLU，在https://arxiv.org/abs/1505.00853中有提到，并
在文章中通过MXNet框架进行实现，在caffe 框架中同样也是没有进行标准定义的。


Permute层实现的功能是将一个多维的Tensor的维度顺序进行切换，例如将一个
NCHW顺序的Tensor转换为NHWC顺序的Tensor。

PSROIPooling层的操作与ROIPooling层类似，不同之处在于不同空间维度输出的图片
特征来自不同的feature map channels，且对每个小区域进行的是Average Pooling，不同
于ROIPooling 的Max Pooling

Upsample层为Pooling层的逆操作，下图为一个示意图，其中每个Upsample层均与网
络之前一个对应大小输入、输出Pooling层一一对应，完成feature map在spatial维度
上的扩充。 
