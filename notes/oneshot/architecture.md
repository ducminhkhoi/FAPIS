#### architecture

#### backbone

ResNet-50

#### RPN

3 anchor ratio: (0.5 1 2)

5 scale: (32 64 125 256 512)   i = {2 ... 6}

3 * 3 * 512 conv   + 1*1 conv 

outputting k times
number of anchors per location (three in our case) features (corresponding to proposal logits
for k = 2 or to bounding box deltas for k = 4).

#### Classification and bounding box regression head

##### input:

output of RPN, croped to 7 * 7

6000 top scoring anchors 才被传入head

##### architecture

两个全连接层，加上一个logistic regression 到2 cat

##### bounding box regression

output 层输出bounding box 修正

##### NMS

threshod = 0.7，

#### segmentation head

##### input

output of rpn, croped to 14*14

##### architecture

4个3*3conv(with relu and **BN**?)  +  1个2 × 2，stride为2的conv + 1 *1 输出层

outputting two feature
maps consisting of logits for foreground/background at each spatial location.