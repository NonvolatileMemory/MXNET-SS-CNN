import mxnet as mx
#导入mxnet
from mxnet import nd
#导入mxnet的矩阵，相当于nmupy矩阵，操作在 mxnet.io里面可以找到
from mxnet.gluon.model_zoo import vision as models
#导入包的地方
from mxnet import image
#读取图片的工具
from mxnet.gluon import nn
#mxnet中有个库叫gluon，gluon中有个nn，这个nn是一个high level api
import csv
#
from mxnet import autograd
#求导的工具
import numpy as np
ctx = mx.gpu(0)
#指定数据模型存放的地方

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])
#vgg读图的均值方差

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32')/255 - rgb_mean) / rgb_std
    return img.transpose((2,0,1)).expand_dims(axis=0)
#vgg读图的预处理，直接用即可，不是我写的

vgg = models.vgg19(pretrained=True)
#调用models拿到预训练的vgg

def get_vgg(vgg):
    net = nn.Sequential()#构建神经网络的high level api用法
    for i in range(37):
        net.add(vgg.features[i])
    return net
#我们需要用到vgg的前37层，挨个添加进来


class MyNet(nn.Block):
    def __init__(self, **kwargs):#初始化方法
        super(MyNet, self).__init__(**kwargs)
        with self.name_scope():
        #以上代码仿写即可，无需了解意思
            self.conv1 = nn.Conv2D(channels= 512,kernel_size=3)
            self.conv2 = nn.Conv2D(channels= 512,kernel_size=3)
            self.conv3 = nn.Conv2D(channels= 512,kernel_size=3)
            self.dense = nn.Dense(2)
    def forward(self, x):#数据流动方法，也即正向传播的方法，x是输入的数据，这种方法比nn.sequential灵活的多，但是略有吗发
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        res = self.dense(x_3)
        return  res

csv_file = []
dajiba = 0
for line in csv.reader(open('/home/user1/zhaozheng/city/csv/votes_safety.csv','r')):
    if(dajiba==0):
        print('bigdick')
        dajiba = 1
    else:
        if(line[2]!='equal'):
            csv_file.append(line)

print("csv read successfule")
def batchGenerator(batch_size):
    for i in range(300000//batch_size):
        left_batch = nd.zeros((batch_size,3,224,224),ctx = ctx)
        right_batch = nd.zeros((batch_size,3,224,224),ctx = ctx)
        label_batch = []
        t = 0
        csv_t = 0
        head_index = batch_size*i
        while(t<batch_size):
            if (csv_file[head_index+csv_t][2] == 'left'):
                label = 1
                left_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + csv_file[head_index+csv_t][0]+ '.jpg')
                right_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + csv_file[head_index+csv_t][1] + '.jpg')
                left_image = preprocess(left_image, (224, 224))
                right_image = preprocess(right_image, (224, 224))
                label_batch.append(label)

                left_batch[t] = left_image[0]
                right_batch[t] = right_image[0]
                #left_batch[t] = nd.reshape(left_image,(3,224,224))
                #right_batch[t] = nd.reshape(right_image,(3,224,224))
                t = t + 1
#                print(t)
            elif (csv_file[head_index+csv_t][2] == 'right'):
                label = 0
                left_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + csv_file[head_index+csv_t][0] + '.jpg')
                right_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + csv_file[head_index+csv_t][1] + '.jpg')
                left_image = preprocess(left_image, (224, 224))
                right_image = preprocess(right_image, (224, 224))
                label_batch.append(label)

                left_batch[t] = left_image[0]
                right_batch[t] = right_image[0]
                #left_batch[t] = nd.reshape(left_image,(1,3,224,224))
                #right_batch[t] = nd.reshape(right_image,(1,3,224,224))
                t = t+1
 #               print(t)
            elif (csv_file[head_index+csv_t][2] == 'equal'):
                print('gui')
                #continue
            csv_t = csv_t + 1
        yield left_batch,right_batch,label_batch

def valGenerator(batch_size):
    for i in range(300000//batch_size):
        left_batch = nd.zeros((batch_size,3,224,224),ctx = ctx)
        right_batch = nd.zeros((batch_size,3,224,224),ctx = ctx)
        label_batch = []
        t = 0
        csv_t = 0
        head_index = batch_size*i
        while(t<batch_size):
            if (csv_file[head_index+csv_t][2] == 'left'):
                label = 1
                left_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + csv_file[head_index+csv_t][0]+ '.jpg')
                right_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + csv_file[head_index+csv_t][1] + '.jpg')
                left_image = preprocess(left_image, (224, 224))
                right_image = preprocess(right_image, (224, 224))
                label_batch.append(label)

                left_batch[t] = left_image[0]
                right_batch[t] = right_image[0]
                #left_batch[t] = nd.reshape(left_image,(3,224,224))
                #right_batch[t] = nd.reshape(right_image,(3,224,224))
                t = t + 1
#                print(t)
            elif (csv_file[head_index+csv_t][2] == 'right'):
                label = 0
                left_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + csv_file[head_index+csv_t][0] + '.jpg')
                right_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + csv_file[head_index+csv_t][1] + '.jpg')
                left_image = preprocess(left_image, (224, 224))
                right_image = preprocess(right_image, (224, 224))
                label_batch.append(label)

                left_batch[t] = left_image[0]
                right_batch[t] = right_image[0]
                #left_batch[t] = nd.reshape(left_image,(1,3,224,224))
                #right_batch[t] = nd.reshape(right_image,(1,3,224,224))
                t = t+1
 #               print(t)
            elif (csv_file[head_index+csv_t][2] == 'equal'):
                print('gui')
                #continue
            csv_t = csv_t + 1
        return left_batch,right_batch,label_batch



def singleImageGenerator():
    for line in csv.reader(open('/home/user1/zhaozheng/city/csv/votes_safety.csv','r')):

        if(line[2] == 'left'):
            label = 1
            left_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + line[0] + '.jpg')
            right_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/' + line[1] + '.jpg')
            left_image = preprocess(left_image,(224,224))
            right_image = preprocess(right_image,(224,224))
        elif(line[2] == 'right'):
            label = 0
            left_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/'+line[0] + '.jpg')
            right_image = image.imread('/home/user1/zhaozheng/city/Places_Pulse/'+line[1] + '.jpg')
            left_image = preprocess(left_image,(224,224))
            right_image = preprocess(right_image,(224,224))
        elif(line[2] == 'equal'):
            continue
        yield left_image,right_image,label

mynet = MyNet()#实例化对象
mynet.initialize(ctx = ctx)#对模型进行初始化，如不指定ctx 则初始化在cpu上面
vgg_extra = get_vgg(vgg)#得到vgg的实力对象
vgg_extra.collect_params().reset_ctx(ctx)#这里有个坑，mxnet默认读取vgg在cpu，你要reset到cpu赏，否则会很慢
softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()#声明loss
trainer = mx.gluon.Trainer(mynet.collect_params() and vgg_extra.collect_params(), 'adam', {'learning_rate': 0.001})#定义optimizer，注意到必须要填写待优化的参数字典，对一个模型对象使用collect——param方法就可以得到，两个字典可以求并

#for a,b,c in valGenerator(500):
a,b,c = valGenerator(64)
test_left = nd.array(a,ctx = ctx)
print(test_left.shape)
test_right = nd.array(b,ctx = ctx)
test_label = nd.array(c,ctx = ctx)#注意，所有的nd。array 初始化的时候也要在同一个context赏
#    break
batch_size = 32

for epoch in range(5):
    bigTiao = 1
    for left,right,label in batchGenerator(batch_size):
        #test_left = []
        #test_right = []
        #test_label = []
        if(bigTiao == 0):
            test_left = left
            test_right = right
            test_label= label
            bigTiao = bigTiao + 1
        with autograd.record():#使用这条语句，记录求导，记住模型训练的输入开始到拿到loss都是要被记录的，除此以外不用记录别的东西
            left_feature = vgg_extra(left)#佐图
            right_feature = vgg_extra(right)#右图
            join_feature = nd.concat(left_feature, right_feature)#连接起来，注意使用nd api
            #join_feature = join_feature.copyto(ctx)
            output = mynet(join_feature)#mxnet.io 放到mxnet里面得到 output
            print(output)
            label = nd.array(label).copyto(ctx)#nd array，copytto指从cpu copy到gpu
            loss = softmax_cross_entropy(output,label)#loss，这个loss也可以手写，保证额可以求导即可
            print(nd.mean(loss))
        loss.backward()#这个就是反向传播，这句话的作用就是bp，注意到，只有在 上面autograd.record()才能bp
        trainer.step(batch_size)
        if(bigTiao == 1):
            #test_left = nd.array(test_left,ctx=ctx)
            #test_right = nd.array(test_right,ctx=ctx)
            test_left_fea = vgg_extra(test_left)
            test_right_fea = vgg_extra(test_right)
            test_label = nd.array(test_label).copyto(ctx)
            test_joint_fea = nd.concat(test_left_fea,test_right_fea)
            test_output = mynet(test_joint_fea)
            acc = mx.metric.Accuracy()#api有，直接计算，ok
            acc.update(preds = [test_output],labels = [test_label])
            print(acc.get())
