import paddle

class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()

        self.resnet = paddle.vision.models.resnet50(pretrained=True)      # 采用Paddle模型库中的ResNet50作为基础模型
        self.fc_retrieval = paddle.nn.Linear(1000, 20)                    # 修改检索输出为20个类

        self.att = paddle.nn.Linear(1000, 1)                              # 定义多视图自适应注意力网络
        self.fc_identify = paddle.nn.Linear(1000, 2)                      # 修改鉴定输出为2个类

    def forward(self, x):
        x = self.resnet(x)
        retrieval = self.fc_retrieval(x)

        att = paddle.transpose(self.att(x), perm=(1, 0))
        x = paddle.matmul(att, x)
        identify = self.fc_identify(x)

        return retrieval, identify