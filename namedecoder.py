import numpy as np
import csv

def onehot(value,classes):
    zerovec=np.zeros(classes)
    if value==-1:
        return zerovec
    #print(value)
    else:
        zerovec[value]=1
        return zerovec

def disease2onehot(diseasename):

    index=0
    str2int={}
    with open('disease4code.csv', 'r') as f:  # 采用b的方式处理可以省去很多问题
        reader = csv.DictReader(f)
        for x in reader:
            str=x['类目编码']
            id=x['id']
            str2int[str]=int(id)
            index=index+1

    charset = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    string = diseasename
    print(string)


    if string=='':
        value=-1
        return np.array([-1])

    elif string[0] in charset: #第一位不是汉字
        value = str2int[string]
        print(value)
    else:
        value=-1 #不详
        return np.array([-1])

    #print(value)

    return onehot(value,index)

def multihot2drugs(dvector):
    int2str = {}
    with open('drugcode.csv', 'r') as f:  # 采用b的方式处理可以省去很多问题
        reader = csv.DictReader(f)
        for row in reader:
            str = row['中成药名称']
            # print(str)
            id = row['id']
            int2str[int(id)] = str
    length=dvector.size
    for i in range(length):
        if dvector[0, i]>0.8:
            print(dvector[0, i])
            print(int2str[i])
