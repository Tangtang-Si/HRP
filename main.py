from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from layers.utils import train,test,my_plot
from layers.dataset import RHPDataset
from transformers import BertTokenizer,logging
from torch.utils.data import DataLoader, SubsetRandomSampler, dataloader
from torch import optim
import numpy as np
import torch
import os
import random
from datetime import datetime
import json
import time
from model.mgan import Mgan


start = datetime.now()
logging.set_verbosity_error()
def set_seed(seed): # 保证每次结果一样,验证是否有用
    if seed != 0:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
###参数设置
EPOCHS = 30
BATCH_SIZE = 4
MAX_LEN = 302
fea_size = 256
num_head = 8
drop = 0.5
scheConf = {'factor':0.2,'patience':5}
optiConf ={'lr':1e-5,'decay':0.05}
SEED = 42
FOLD_NUM = 10
MODEL='mgan'
set_seed(seed=SEED)
output_pth = f'output/{MODEL}/{MODEL}'+start.strftime('%m%d%H')
json_pth = output_pth+".json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {'EPOCHS':EPOCHS,'BATCH_SIZE':BATCH_SIZE,'MAX_LEN':MAX_LEN,
        'fea_size':fea_size,'num_head':num_head,'drop':drop,
          'scheConfig':scheConf, 'optiConfig':optiConf}

target_file = f'./output/mgan_best.pth.tar'

### 模型训练
print(20 * "=", " Preparing for training ", 20 * "=")
# 加载数据集
tokenizer = BertTokenizer.from_pretrained('./data/bert-base-uncased')
all_data = RHPDataset(fname='./data/experiment.xlsx',max_length = MAX_LEN,tokenizer=tokenizer)
train_data,test_data = train_test_split(all_data,train_size=0.7,shuffle=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=False)

### 定义模型参数
model_dict = {
    'mgan': {'model': Mgan,
             'inputs': ['label','input_ids','attention_mask','image_input'],
             'inits': {'fea_size':fea_size,'num_head':num_head,'drop':drop}}
}
# # 不用十折交叉验证
# print("Building {} model...".format(MODEL))
# inputs = model_dict[MODEL]['inputs']
# model = model_dict[MODEL]['model'](model_dict[MODEL]['inits'])
# model = model.cuda()
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=optiConf['lr'], weight_decay=optiConf['decay'])
# start_epoch = 1
# best_score = 0.0
# history = {'log':config,'train_loss': [], 'train_acc': [],'train_rec':[],'train_pre':[],'train_f1':[],
#            'train_auc':[],'test_loss': [], 'test_acc': [],'test_rec':[],'test_pre':[],'test_f1':[],'test_auc':[],
#            'dev_loss': [], 'dev_acc': [],'dev_rec':[],'dev_pre':[],'dev_f1':[],'dev_auc':[]}
#
# # 模型训练
# print("\n", 20 * "=", "Training model on device: {}".format(device), 20 * "=")
# patience_counter = 0
# for epoch in range(start_epoch, EPOCHS + 1):
#     print("\t* Training fo epoch [{}/{}]:".format(epoch, EPOCHS))
#     loss, accuracy, recall,precision, f1_score, auc = train(device,inputs,model,criterion,optimizer,train_loader,epoch)
#     history['train_loss'].append(loss)
#     history['train_acc'].append(accuracy)
#     history['train_rec'].append(recall)
#     history['train_pre'].append(precision)
#     history['train_f1'].append(f1_score)
#     history['train_auc'].append(auc)
#     print("Train: loss {0:.4f};Accuracy {1:.4f}; F1_Score {2:.4f};auc{3:.4f}".format(loss, accuracy, f1_score,auc))
#
#     loss, accuracy, recall,precision, f1_score, auc= test(device,inputs,model,criterion,test_loader,epoch)
#     history['test_loss'].append(loss)
#     history['test_acc'].append(accuracy)
#     history['test_rec'].append(recall)
#     history['test_pre'].append(precision)
#     history['test_f1'].append(f1_score)
#     history['test_auc'].append(auc)
#     print("Test: loss {0:.4f};Accuracy {1:.4f};F1_Score {2:.4f};auc{3:.4f}".format(loss,accuracy, f1_score,auc))
#
#     # Early stopping on validation accuracy.
#     if accuracy < best_score:
#         patience_counter += 1
#     else:
#         print('save data')
#         best_score = accuracy
#         patience_counter = 0
#         torch.save({"epoch": epoch,
#                     "model": model.state_dict(),
#                     "best_score": best_score,
#                     "history": history},
#                      target_file )
#     if patience_counter >= 20:
#         print("-> Early stopping: patience limit reached, stopping...")
#         break
#
# with open('output/mgan/json_pth',"w",encoding='utf-8') as f:
#     f.write(json.dumps(history,ensure_ascii=False,indent=4))
#
# my_plot(history['train_acc'], history['test_acc'], history['train_loss'])
# print(best_score)


### 采用十折交叉验证完整代码如下：
print("Building {} model...".format(MODEL))
fold_dic={}
fold_dic['config'] = config  # 保存实验参数
splits = StratifiedKFold(n_splits=FOLD_NUM,shuffle=True) # 分层10折交叉验证
x = np.arange(all_data.__len__())  # 输入的idx
y = np.array(all_data.getLabel())  # 输入的label
inputs = model_dict[MODEL]['inputs']
# # 十折交叉验证
for fold, (train_idx, val_idx) in enumerate(splits.split(x, y)):
    print('Fold {}'.format(fold + 1))
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(all_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_loader = DataLoader(all_data, batch_size=BATCH_SIZE, sampler=test_sampler)
    model = model_dict[MODEL]['model'](model_dict[MODEL]['inits'])
    model = model.cuda(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=optiConf['lr'],
                           weight_decay=optiConf['decay'])  # adam和adamW（过拟合）的区别！！！
    # optimizer = optim.SGD(model.parameters(),lr=optiConf['lr'],weight_decay=optiConf['decay'])# for image,效率低下
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=scheConf['factor'],
                                                     patience=scheConf['patience'],
                                                     verbose=False, threshold=0.01, threshold_mode='rel', cooldown=0,
                                                     min_lr=0, eps=1e-8)
    history = {'train_loss': [], 'train_acc': [], 'train_rec': [], 'train_pre': [], 'train_f1': [], 'train_auc': [],
               'test_loss': [], 'test_acc': [], 'test_rec': [], 'test_pre': [], 'test_f1': [], 'test_auc': []}

    for epoch in range(EPOCHS):
        print("%d epoch..." % epoch)
        train_loss, train_acc, train_rec,train_pre,train_f1, train_auc = train(device,inputs,model,criterion,optimizer,train_loader,epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_rec'].append(train_rec)
        history['train_pre'].append(train_pre)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)
        test_loss, test_acc,test_rec,test_pre, test_f1, test_auc = test(device,inputs,model,criterion,test_loader,epoch)
        scheduler.step(test_auc)

        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_rec'].append(test_rec)
        history['test_pre'].append(test_pre)
        history['test_f1'].append(test_f1)
        history['test_auc'].append(test_auc)
    fold_dic['fold{}'.format(fold + 1)] = history

### 文件保存
with open(json_pth, "w", encoding='utf-8') as f:  # 设置'utf-8'编码
    f.write(json.dumps(fold_dic, ensure_ascii=False, indent=4))  # # 如果ensure_ascii=True则会输出中文的ascii码，这里设为False

### 结果分析
### 文件读取
with open('output/mgan/mgan031617.json','r',encoding = 'utf-8') as f:  # json_pth导入模型数据进行分析
    fold_dic = json.load(f)


# 计算每个fold所有epoch下的最大值求平均
train_loss,train_acc,train_rec, train_pre,train_f1,train_auc =[],[],[],[],[],[]
test_loss,test_acc,test_rec, test_pre,test_f1,test_auc =[],[],[],[],[],[]
for f in range(1,FOLD_NUM+1):
    m_index = np.array(fold_dic['fold{}'.format(f)]['test_acc']).argmax()
    train_loss.append(fold_dic['fold{}'.format(f)]['train_loss'][m_index])
    train_acc.append(fold_dic['fold{}'.format(f)]['train_acc'][m_index])
    train_rec.append(fold_dic['fold{}'.format(f)]['train_rec'][m_index])
    train_pre.append(fold_dic['fold{}'.format(f)]['train_pre'][m_index])
    train_f1.append(fold_dic['fold{}'.format(f)]['train_f1'][m_index])
    train_auc.append(fold_dic['fold{}'.format(f)]['train_auc'][m_index])
    test_loss.append(fold_dic['fold{}'.format(f)]['test_loss'][m_index])
    test_acc.append(fold_dic['fold{}'.format(f)]['test_acc'][m_index])
    test_rec.append(fold_dic['fold{}'.format(f)]['test_rec'][m_index])
    test_pre.append(fold_dic['fold{}'.format(f)]['test_pre'][m_index])
    test_f1.append(fold_dic['fold{}'.format(f)]['test_f1'][m_index])
    test_auc.append(fold_dic['fold{}'.format(f)]['test_auc'][m_index])
print('config of the model:{}'.format(fold_dic['config']))
print("Average Train: loss {0:.4f};Accuracy {1:.4f};Recall {2:.4f};Precision {3:.4f}; F1_Score {4:.4f};auc{5:.4f} \
    ".format(np.mean(train_loss),np.mean(train_acc),np.mean(train_rec),np.mean(train_pre),np.mean(train_f1),np.mean(train_auc)))
print("Average Test: loss {0:.4f};Accuracy {1:.4f};Recall {2:.4f};Precision {3:.4f}; F1_Score {4:.4f};auc{5:.4f} \
    ".format(np.mean(test_loss),np.mean(test_acc),np.mean(test_rec),np.mean(test_pre),np.mean(test_f1),np.mean(test_auc)))

# # 每个epoch对所有fold求平均E
# train_loss,train_acc,train_rec,train_pre,train_f1,train_auc =[],[],[],[],[],[]
# test_loss,test_acc,test_rec,test_pre,test_f1,test_auc =[],[],[],[],[],[]
# for i in range(EPOCHS):
#     ftrain_loss,ftrain_acc,ftrain_rec,ftrain_pre,ftrain_f1,ftrain_auc =[],[],[],[],[],[]
#     ftest_loss,ftest_acc,ftest_rec,ftest_pre,ftest_f1,ftest_auc =[],[],[],[],[],[]
#     for f in range(1,FOLD_NUM+1):
#         ftrain_loss.append(fold_dic['fold{}'.format(f)]['train_loss'][i])
#         ftrain_acc.append(fold_dic['fold{}'.format(f)]['train_acc'][i])
#         ftrain_rec.append(fold_dic['fold{}'.format(f)]['train_rec'][i])
#         ftrain_pre.append(fold_dic['fold{}'.format(f)]['train_pre'][i])
#         ftrain_f1.append(fold_dic['fold{}'.format(f)]['train_f1'][i])
#         ftrain_auc.append(fold_dic['fold{}'.format(f)]['train_auc'][i])
#         ftest_loss.append(fold_dic['fold{}'.format(f)]['test_loss'][i])
#         ftest_acc.append(fold_dic['fold{}'.format(f)]['test_acc'][i])
#         ftest_rec.append(fold_dic['fold{}'.format(f)]['test_rec'][i])
#         ftest_pre.append(fold_dic['fold{}'.format(f)]['test_pre'][i])
#         ftest_f1.append(fold_dic['fold{}'.format(f)]['test_f1'][i])
#         ftest_auc.append(fold_dic['fold{}'.format(f)]['test_auc'][i])
#     train_loss.append(np.mean(ftrain_loss))
#     train_acc.append(np.mean(ftrain_acc))
#     train_rec.append(np.mean(ftrain_rec))
#     train_pre.append(np.mean(ftrain_pre))
#     train_f1.append(np.mean(ftrain_f1))
#     train_auc.append(np.mean(ftrain_auc))
#     test_loss.append(np.mean(ftest_loss))
#     test_acc.append(np.mean(ftest_acc))
#     test_rec.append(np.mean(ftest_rec))
#     test_pre.append(np.mean(ftest_pre))
#     test_f1.append(np.mean(ftest_f1))
#     test_auc.append(np.mean(ftest_auc))

plt.rcParams["font.sans-serif"]=["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"]=False
epoch = range(EPOCHS)
plt.figure(figsize=(10,8))
plt.plot(epoch,train_loss,label='train_loss')
plt.plot(epoch,test_loss,label = 'test_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss Variation')
plt.show()

# # 模型配置
# # 数据集正负标签数量获取
# x=0
# y=0
# for i in range(len(train_data)):
#     if train_data[i]['label']==1:
#         x+=1
# for i in range(len(dev_data)):
#     if dev_data[i]['label']==1:
#         y+=1
#


# # 保存数据
# with open("data/train.pkl", "wb") as f:
#     pickle.dump(train_data, f)
# # fr = open("dataset/train.pkl",'rb')# open的参数是pkl文件的路径
# # train_data = pickle.load(fr)  # 读取pkl文件的内容
#
# with open("data/dev.pkl", "wb") as f:
#     pickle.dump(dev_data, f)
# # fd = open("dataset/dev.pkl",'rb')# open的参数是pkl文件的路径
# # dev_data = pickle.load(fd)  # 读取pkl文件的内容
#
# my_plot(history['train_acc'], history['test_acc'], history['train_loss'])



# # 保存模型方式2（模型参数）相应加载模型方式
# module = torch.load('output/mgan/mgan_best.pth.tar')
# model = Mgan(model_dict[MODEL]['inits'])
# model = Mgan(module)
# model = model.cuda()
# loss, accuracy = test(dev_loader, model,5,device)
# print("Validation: loss {0:.4f};Accuracy {1:.4f}".format(loss, accuracy))