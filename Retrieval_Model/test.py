import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import loadF,split_dataset
from model import TestModel
import time
from re_ranking import re_ranking

def test(model,database_img,database_label,query_img,query_label,K):
    model_path = './Model.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.eval().cuda()
    average_precision_li = []
    with torch.no_grad():
        database_feat=model(database_img.cuda())
        query_feat=model(query_img.cuda())
        for i in tqdm(range(0, len(query_img)), desc='Computing', total=len(query_img)):
            label=query_label[i]
            query=query_feat[i].expand(database_feat.shape)
            sim = F.cosine_similarity(database_feat, query)
            _, indices = torch.topk(sim, K)
            match_list = database_label[indices] == label
            pos_num = 0
            total_num = 0
            precision_li = []
            for item in match_list:
                if item == 1:
                    pos_num += 1
                    total_num += 1
                    precision_li.append(pos_num / float(total_num))
                else:
                    total_num += 1
            if not precision_li:
                average_precision_li.append(0)
            else:
                average_precision = np.mean(precision_li)
                average_precision_li.append(average_precision)

        mAP = np.mean(average_precision_li)
    print(f'test mAP@{K}: {mAP}')

def test_stage(load_model, Xp_test, yp_test):
    model_path = './Model.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model.load_state_dict(checkpoint)
    load_model.eval().cuda()
    with torch.no_grad():
        pred_y = load_model(Xp_test.cuda())
        # print(pred_y)
        average_precision_li = []
        for idx in range(len(yp_test)):
            query = pred_y[idx].expand(pred_y.shape)
            label = yp_test[idx]
            sim = F.cosine_similarity(pred_y, query)
            _, indices = torch.topk(sim, 300)
            match_list = yp_test[indices] == label
            pos_num = 0
            total_num = 0
            precision_li = []
            for item in match_list[1:]:
                if item == 1:
                    pos_num += 1
                    total_num += 1
                    precision_li.append(pos_num / float(total_num))
                else:
                    total_num += 1
            if not precision_li:
                average_precision_li.append(0)
            else:
                average_precision = np.mean(precision_li)
                average_precision_li.append(average_precision)
        mAP = np.mean(average_precision_li)
    print(f'test mAP: {mAP}')

def cpu_test(model,data):
    model_path = './Model_10K.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.eval()
    data=torch.from_numpy(np.array(data)).to(torch.float32)
    start=time.time()
    with torch.no_grad():
        result=model(data)
    end=time.time()
    print(end-start)

def topKacc(model,database_img,database_label,query_img,query_label,K):
    model_path = './Model_10K.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.eval().cuda()
    average_precision_li = []
    with torch.no_grad():
        database_feat=model(database_img.cuda())
        query_feat=model(query_img.cuda())
        for i in tqdm(range(0, len(query_img)), desc='Computing', total=len(query_img)):
            label=query_label[i]
            query=query_feat[i].expand(database_feat.shape)
            sim = F.cosine_similarity(database_feat, query)
            _, indices = torch.topk(sim, K)
            match_list = database_label[indices] == label
            hit = 0
            for item in match_list:
                if item == 1:
                    hit += 1
            average_precision = hit/K
            average_precision_li.append(average_precision)

        mAP = np.mean(average_precision_li)
    print(f'Top {K} Acc: {mAP}')

def Reranking_mAp(dist,gallery_label,query_label,topK=100):
    query_num=len(query_label)
    average_precision_li = []
    dist=torch.tensor(dist)
    for i in tqdm(range(0, query_num), desc='Computing', total=query_num):
        label = query_label[i]
        _, indices = torch.topk(dist[i], topK, largest=False)
        match_list = gallery_label[indices] == label
        pos_num = 0
        total_num = 0
        precision_li = []
        for item in match_list:
            if item == 1:
                pos_num += 1
                total_num += 1
                precision_li.append(pos_num / float(total_num))
            else:
                total_num += 1
        if precision_li == []:
            average_precision_li.append(0)
        else:
            average_precision = np.mean(precision_li)
            average_precision_li.append(average_precision)
    mAP = np.mean(average_precision_li)
    return mAP

def temp(Xp_train, yp_train, Xp_test, yp_test):
    model=TestModel(word_dim=64,n_blocks=8,n_classes=100,represent_dim=128)
    checkpoint = torch.load('./Model_10K.pth')
    model.load_state_dict(checkpoint)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    start=time.time()
    with torch.no_grad():
        gallery_f = model(Xp_train.cuda())
        query_f = model(Xp_test.cuda())
    dist = re_ranking(query_f,gallery_f)
    print(Reranking_mAp(dist,yp_train,yp_test))
    end=time.time()
    print(end-start)

if __name__=='__main__':
    IMG_NUM=10000
    Features=loadF('../data/features/DCTHistfeats_10K.csv',IMG_NUM)
    Labels=np.load('../data/Label/Label_10K.npy')
    y=np.zeros(IMG_NUM)
    for i in range(IMG_NUM):
        y[i]=Labels[i][0]
    Xp_train, yp_train, Xp_test, yp_test = split_dataset(Features,y)
    model=TestModel(word_dim=64,n_blocks=8,n_classes=100,represent_dim=128)

    device = torch.device('cuda:0')
    model.to(device)
    
    #temp(Xp_train, yp_train, Xp_test, yp_test)
    #test_stage(model, Xp_test, yp_test)
    #test(model,Xp_train,yp_train,Xp_test,yp_test,100)
    #cpu_test(model,Features)
    topKacc(model,Xp_train,yp_train,Xp_test,yp_test,30)