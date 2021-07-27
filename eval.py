import paddle
from tqdm import tqdm

import argparse

from dataset import *
from model import *
from sklearn.metrics import f1_score, precision_score, recall_score

parse = argparse.ArgumentParser(description="ysneaker predictor model")
parse.add_argument('--dataset_path', help='images dataset path', type=str)
parse.add_argument('--identify_path_test', help='identify dataset path', type=str)
parse.add_argument('--retrieval_path_test', help='retrieval dataset path', type=str)
parse.add_argument('--batch_size', help='data load batch size', type=int)
args = parse.parse_args()

def test(model, args):

    batch_size = args.batch_size
    dataset_test = MyDataset(args, split='test')                           # 加载测试数据
    loader_test = paddle.io.DataLoader(dataset_test,                       # 将数据加载进paddle dataloader中，并设置batch_sizebatch_size
                                       batch_size=batch_size,
                                       shuffle=False)

    model.eval()
    test_result = []
    test_true_result = []
    test_result_ident = []
    test_true_result_ident = []
    for batch_test_idx, batch_test in enumerate(tqdm(loader_test())):
        data_test, label_test = batch_test[0].cuda(), batch_test[1].cuda()
        data_ident_test_x, label_ident_test_y = batch_test[2].cuda(), batch_test[3].cuda()

        with paddle.no_grad():
            retrieval_test = model(data_test)[0]
            softmax = paddle.nn.Softmax()
            output_list = softmax(retrieval_test)
            output_list = paddle.argmax(output_list, axis=1)
            output_list = paddle.tolist(output_list)
            test_result.extend(output_list)

            tmp_identify_eval = paddle.empty((1, 2)).cuda()
            for j in range(batch_size):
                identify_eval = model(data_ident_test_x[j])[1]
                tmp_identify_eval = paddle.concat((tmp_identify_eval, identify_eval), 0)

            output_list_identify = softmax(tmp_identify_eval[1:])
            output_list_identify = paddle.argmax(output_list_identify, axis=1)
            output_list_identify = paddle.tolist(output_list_identify)
            test_result_ident.extend(output_list_identify)

        label_val_list = paddle.tolist(label_test)
        test_true_result.extend(label_val_list)
        label_val_list_ident = paddle.tolist(label_ident_test_y)
        test_true_result_ident.extend(label_val_list_ident)

    F1 = f1_score(test_result, test_true_result, average='macro')
    print(test_result)
    Precision = precision_score(test_result, test_true_result, average='macro')
    Recall = recall_score(test_result, test_true_result, average='macro')
    print("Result_retrieval F1 Score Macro: {}, Precision score: {}, Recall score: {}".format(F1,
                                                                                              Precision,
                                                                                              Recall))

    identify_F1 = f1_score(test_result_ident, test_true_result_ident[:len(test_result_ident)], average='binary')
    print(test_result_ident, test_true_result_ident[:len(test_result_ident)])
    identify_Precision = precision_score(test_result_ident, test_true_result_ident[:len(test_result_ident)], average='binary')
    identify_Recall = recall_score(test_result_ident, test_true_result_ident[:len(test_result_ident)], average='binary')
    print("Result_identify F1 Score Macro: {}, Precision score: {}, Recall score: {}".format(identify_F1,
                                                                                             identify_Precision,
                                                                                             identify_Recall))

    return test_result, test_result_ident


if __name__=='__main__':

    # test
    model = MyNet()                                                   # 初始化模型网络
    model.set_state_dict(paddle.load('./model/model.pdparams'))       # 加载预训练模型
    result_retrieval, result_identify = test(model, args)             # 测试

    # label字典
    brand2label = {
                 0: 'Nike',
                 1: 'Adidas',
                 2: 'air jordan 1',
                 3: 'Vans',
                 4: 'air jordan 11',
                 5: 'Converse',
                 6: 'Puma',
                 7: 'air jordan 6',
                 8: 'New Balance',
                 9: 'air jordan 4',
                 10: 'Asics',
                 11: 'air jordan 5',
                 12: 'air jordan 3',
                 13: 'air jordan 13',
                 14: 'Haven',
                 15: 'air jordan 7',
                 16: 'air jordan 32',
                 17: 'air jordan 12',
                 18: 'Reebok',
                 19: 'Under Armour'
    }

    true_false = {
        0: "False",
        1: 'True'
    }

    # 结果输出
    for index in result_retrieval:
        print("Sneaker brand: {}".format(brand2label[index]))

    for index in result_identify:
        print("{} Sneaker".format(true_false[index]))