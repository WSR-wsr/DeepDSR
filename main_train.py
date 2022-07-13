import numpy as np
import pandas as pd
from eff_model import train as effv2_train
from eff_model import predict as effv2_predict
from resnet_model import train as resnet_train
from resnet_model import predict as resnet_predict
from vit_model import train as vit_train
from vit_model import predict as vit_predict
from swin_model import train as swin_train
from swin_model import predict as swin_predict
from reg_model import train as reg_train
from reg_model import predict as reg_predict
from dense_model import train as dense_train
from dense_model import predict as dense_predict
from utils import read_data
import os


def mk_dir(j):
    if os.path.exists('./res_dir/label/data' + str(j)) is False:
        os.makedirs('./res_dir/label/data' + str(j))
    if os.path.exists('./res_dir/train_res/data' + str(j)) is False:
        os.makedirs('./res_dir/train_res/data' + str(j))
    if os.path.exists('./res_dir/weights/data' + str(j)) is False:
        os.makedirs('./res_dir/weights/data' + str(j))
    if os.path.exists('./res_dir/best_res') is False:
        os.makedirs('./res_dir/best_res')


val = False
TRAIN = True
PREDICT = True
PRE_train = True
n_class = 2
ep = 100
device = 'cuda:0'
if __name__ == '__main__':
    for i in range(1, 2):
        data_path = './data/data' + str(i) + '.txt'
        mk_dir(i)
        train, train_label, test, test_label = read_data(data_path, 0.2, n_class=n_class, seed=2)
        if TRAIN:
            # effv2_train(train, train_label, classes=n_class, device=device, val=False, data=i, epochs=ep, init=PRE_train)
            resnet_train(train, train_label, classes=n_class, device=device, val=False, data=i, epochs=ep, init=PRE_train)
            vit_train(train, train_label, classes=n_class, device=device, val=False, data=i, epochs=ep, init=PRE_train)
            swin_train(train, train_label, classes=n_class, device=device, val=False, data=i, epochs=ep, init=PRE_train)
            reg_train(train, train_label, classes=n_class, device=device, val=False, data=i, epochs=ep, init=PRE_train)
            dense_train(train, train_label, classes=n_class, device=device, val=False, data=i, epochs=ep, init=PRE_train)
            print('train over')
        if PREDICT:
            score1 = effv2_predict(test, test_label, num_class=n_class, data=i)
            score2 = resnet_predict(test, test_label, num_class=n_class, data=i)
            score3 = vit_predict(test, test_label, num_class=n_class, data=i)
            score4 = swin_predict(test, test_label, num_class=n_class, data=i)
            score5 = reg_predict(test, test_label, num_class=n_class, data=i)
            score6 = dense_predict(test, test_label, num_class=n_class, data=i)
            pd.DataFrame(np.array(test_label)).to_csv('./res_dir/label.csv', header=None, index=None)
            print('predict over')
