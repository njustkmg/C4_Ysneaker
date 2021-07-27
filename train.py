import paddle
from tqdm import tqdm

import os
import argparse

from dataset import *
from model import *
from sklearn.metrics import f1_score, precision_score, recall_score

parse = argparse.ArgumentParser(description="ysneaker predictor model")
parse.add_argument('--dataset_path', help='images dataset path', type=str)
parse.add_argument('--identify_path_train', help='identify dataset path', type=str)
parse.add_argument('--retrieval_path_train', help='retrieval dataset path', type=str)
parse.add_argument('--identify_path_val', help='identify dataset path', type=str)
parse.add_argument('--retrieval_path_val', help='retrieval dataset path', type=str)
parse.add_argument('--batch_size', help='data load batch size', type=int)
parse.add_argument('--epoch', help='Epoch train', type=int)
args = parse.parse_args()

def train(args):

    dataset_train = MyDataset(args, split='train')                          # 加载数据
    dataset_eval = MyDataset(args, split='val')

    start_result = 0
    batch_size = args.batch_size

    loader_train = paddle.io.DataLoader(dataset_train,                       # 将数据加载进paddle dataloader中，并设置batch_sizebatch_size
                                        batch_size=batch_size,
                                        shuffle=True)
    loader_val = paddle.io.DataLoader(dataset_eval,
                                      batch_size=batch_size,
                                      shuffle=True)

    model = MyNet()                                                          # 初始化模型网络
    criterion = paddle.nn.CrossEntropyLoss()                                 # 定义损失函数
    optimizer = paddle.optimizer.Adam(learning_rate=1e-5, parameters=model.parameters())        # 定义优化器

    # 训练
    model.train()
    for epoch in range(args.epoch):
        print("Epoch: {}".format(epoch))

        for batch_idx, batch in enumerate(tqdm(loader_train())):
            data, label = batch[0].cuda(), batch[1].cuda()
            data_identify, label_identify = batch[2].cuda(), batch[3].cuda()

            retrieval = model(data)[0]
            loss_retrieval = criterion(retrieval, label)

            tmp_identify = paddle.empty((1, 2)).cuda()
            for ident in range(batch_size):
                identify = model(data_identify[ident])[1]
                tmp_identify = paddle.concat((tmp_identify, identify), 0)

            loss_identify = criterion(tmp_identify[1:], label_identify)

            loss = loss_retrieval + loss_identify
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            # 验证
            if batch_idx % 5000 == 0 and batch_idx != 0:

                model.eval()
                val_result_eval = []
                true_result_eval = []
                val_result_ident = []
                true_result_ident = []
                for batch_val_idx, batch_val in enumerate(tqdm(loader_val())):
                    data_val, label_val = batch_val[0].cuda(), batch_val[1].cuda()
                    data_ident_val_x, label_ident_val_y = batch_val[2].cuda(), batch_val[3].cuda()

                    with paddle.no_grad():
                        retrieval_eval = model(data_val)[0]
                        softmax = paddle.nn.Softmax()
                        output_list = softmax(retrieval_eval)
                        output_list = paddle.argmax(output_list, axis=1)
                        output_list = paddle.tolist(output_list)
                        val_result_eval.extend(output_list)

                        tmp_identify_eval = paddle.empty((1, 2)).cuda()
                        for j in range(batch_size):
                            identify_eval = model(data_ident_val_x[j])[1]
                            tmp_identify_eval = paddle.concat((tmp_identify_eval, identify_eval), 0)

                        output_list_identify = softmax(tmp_identify_eval[1:])
                        output_list_identify = paddle.argmax(output_list_identify, axis=1)
                        output_list_identify = paddle.tolist(output_list_identify)
                        val_result_ident.extend(output_list_identify)

                    label_val_list = paddle.tolist(label_val)
                    true_result_eval.extend(label_val_list)
                    label_val_list_ident = paddle.tolist(label_ident_val_y)
                    true_result_ident.extend(label_val_list_ident)

                F1 = f1_score(val_result_eval, true_result_eval, average='macro')
                Precision = precision_score(val_result_eval, true_result_eval)
                Recall = recall_score(val_result_eval, true_result_eval)
                print("Result_retrieval F1 Score Macro: {}, Precision score: {}, Recall score: {}".format(F1, Precision, Recall))
                identify_F1 = f1_score(val_result_ident, true_result_ident[:len(val_result_ident)], average='macro')
                identify_Precision = precision_score(val_result_eval, true_result_eval)
                identify_Recall = recall_score(val_result_eval, true_result_eval)
                print("Result_identify F1 Score Macro: {}, Precision score: {}, Recall score: {}".format(identify_F1, identify_Precision, identify_Recall))

                if identify_F1 > start_result:
                    start_result = identify_F1
                    model_path = './model'
                    isExists = os.path.exists(model_path)
                    if not isExists:
                        os.makedirs(model_path)
                    checkpoint_path = os.path.join(model_path, 'model_%s.pdparams' % (epoch))
                    paddle.save(model.state_dict(), checkpoint_path)
                    print("Save model_{}".format(epoch))

if __name__=='__main__':

    # train
    train(args)
