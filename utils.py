
from re import A
from cv2 import log
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.image as img
import matplotlib.ticker as ticker
import scipy.io
import random
import math
from scipy.interpolate import make_interp_spline
import os
import seaborn as sns
def bound4temporal_cor(arr):
    total = np.sum(arr)
    lower = 0
    upper = 499
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
        if (i == 0) & (sum > total * 0.05):
            break
        if(sum > total * 0.05):
            lower = i
            break
    sum = 0
    for i in range(len(arr)):
        sum += arr[499-i]
        if(sum > total * 0.05):
            upper = 499-i
            break
    return lower, upper

def plot_cdf():
    plt.rcParams['pdf.fonttype'] = 42
    arr = np.loadtxt('global6')
    a = [i for i in range(50, 2550, 50)]
    
    T = np.array(a)
    
    xnew = np.linspace(T.min(),T.max(),3000)
    # for i in range(len(arr)):
        
    #     tmp = arr[i]
    #     if tmp == 0:
    #         arr[i] = 1
        # arr[i] = math.log(tmp,10)
    print(arr)
    power_smooth = make_interp_spline(T,arr)(xnew)

    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(a, arr, c = 'red', s = 10)
    plt.plot(xnew, arr_positive(power_smooth))
    plt.xlabel("Number of Searched Frames",fontsize=15)
    plt.ylabel("Number of Corresponding Queries",fontsize=15)
    plt.savefig('cdf.pdf')
    # print(arr)

def arr_positive(arr):
    for i in range(len(arr)):
        if arr[i] < 1:
            arr[i] = 1
    return arr

def print_temporal():
    plt.rcParams['pdf.fonttype'] = 42
    model_path = './correlation_split8_0.mat'
    result = scipy.io.loadmat(model_path)
    a = [i for i in range(0, 120,8)]
    T = np.array(a)
    plt.figure(figsize=(10,9),dpi=100)
    xnew = np.linspace(T.min(),T.max(),300)
    

    temporal_distribution = result['temporal']
    power_smooth1 = make_interp_spline(T,temporal_distribution[1][0][:15])(xnew)
    # print(power_smooth1)
    # tmp = arr_positive(power_smooth1)
    # print(tmp)
    power_smooth2 = make_interp_spline(T,temporal_distribution[2][0][:15])(xnew)
    power_smooth3 = make_interp_spline(T,temporal_distribution[3][0][:15])(xnew)
    power_smooth4 = make_interp_spline(T,temporal_distribution[4][0][:15])(xnew)
    power_smooth5 = make_interp_spline(T,temporal_distribution[5][0][:15])(xnew)
    plt.plot(xnew,arr_positive(power_smooth1), color = "red", label = "c1_to_c2")
    plt.plot(xnew,arr_positive(power_smooth2), color = "blue", label = "c1_to_c3")
    plt.plot(xnew,arr_positive(power_smooth3), color = "green", label = "c1_to_c4")
    plt.plot(xnew,arr_positive(power_smooth4), color = "gold", label = "c1_to_c5")
    plt.plot(xnew,arr_positive(power_smooth5), color = "purple", label = "c1_to_c6")
    #plt.plot(range(30), gauss_smooth(temporal_distribution[1][0][:30]), color = "red", label = "c1_to_c2")
    # plt.plot(range(30), temporal_distribution[2][0][:30], color = "blue", label = "c1_to_c3")
    # plt.plot(range(30), temporal_distribution[3][0][:30], color = "green", label = "c1_to_c4")
    # plt.plot(range(30), temporal_distribution[4][0][:30], color = "gold", label = "c1_to_c5")
    # plt.plot(range(30), temporal_distribution[5][0][:30], color = "violet", label = "c1_to_c6")
    plt.xticks(fontproperties = 'Times New Roman',fontsize=30)
    plt.yticks(fontproperties = 'Times New Roman',fontsize=30)
    plt.legend(fontsize=30,loc='best')
    plt.xlabel("Traveling Time (s)",fontsize=30)
    plt.ylabel("Frequency",fontsize=30)
    plt.savefig('temporal.pdf')

def print_spatial():
    plt.rcParams['pdf.fonttype'] = 42
    ax1 = plt.subplot(1,1, 1)
    model_path = './correlation_split8_0.mat'
    result = scipy.io.loadmat(model_path)

    spatial_distribution = result['spatial']
    x_label = ["c1","c2","c3","c4","c5","c6","Exit"]
    y_label = ["c1","c2","c3","c4","c5","c6"]
    ax = sns.heatmap(spatial_distribution, linewidth=0.5, annot=True, cmap="YlGnBu", xticklabels=x_label, yticklabels=y_label)
    # ax.set_title('Spatial Correlation')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
    # ax.figure.axes[-1].yaxis.label.set_size(100)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=24)
    # plt.show()
    # ax.set_xlabel(..., fontsize=15)
    # ax.set_ylabel(..., fontsize=15)
    s = ax.get_figure()
    s.savefig('./spatial.pdf', dpi = 300, bbox_inches = 'tight')



def print_4spatial():
    plt.rcParams['pdf.fonttype'] = 42
    ax1 = plt.subplot(2,2, 1)
    model_path = './correlation_split4_0.mat'
    result = scipy.io.loadmat(model_path)

    spatial_distribution = result['spatial']
    x_label = ["c1","c2","c3","c4","c5","c6","Exit"]
    y_label = ["c1","c2","c3","c4","c5","c6"]
    ax = sns.heatmap(spatial_distribution, linewidth=0.5, annot=True, annot_kws={"fontsize":6}, cmap="YlGnBu", xticklabels=x_label, yticklabels=y_label)
    ax.set_title('Period 1',fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    # plt.show()
    # ax.set_xlabel(..., fontsize=15)
    # ax.set_ylabel(..., fontsize=15)
    ax1 = plt.subplot(2,2, 2)
    model_path = './correlation_split4_1.mat'
    result = scipy.io.loadmat(model_path)

    spatial_distribution = result['spatial']
    x_label = ["c1","c2","c3","c4","c5","c6","Exit"]
    y_label = ["c1","c2","c3","c4","c5","c6"]
    ax = sns.heatmap(spatial_distribution, linewidth=0.5, annot=True, annot_kws={"fontsize":6},fmt ='0.2g', cmap="YlGnBu", xticklabels=x_label, yticklabels=y_label)
    ax.set_title('Period 2',fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)

    ax1 = plt.subplot(2,2, 3)
    model_path = './correlation_split4_2.mat'
    result = scipy.io.loadmat(model_path)

    spatial_distribution = result['spatial']
    x_label = ["c1","c2","c3","c4","c5","c6","Exit"]
    y_label = ["c1","c2","c3","c4","c5","c6"]
    ax = sns.heatmap(spatial_distribution, linewidth=0.5, annot=True, annot_kws={"fontsize":6},fmt ='0.2g', cmap="YlGnBu", xticklabels=x_label, yticklabels=y_label)
    ax.set_title('Period 3',fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)

    ax1 = plt.subplot(2,2, 4)
    model_path = './correlation_split4_3.mat'
    result = scipy.io.loadmat(model_path)

    spatial_distribution = result['spatial']
    x_label = ["c1","c2","c3","c4","c5","c6","Exit"]
    y_label = ["c1","c2","c3","c4","c5","c6"]
    ax = sns.heatmap(spatial_distribution, linewidth=0.5, annot=True, annot_kws={"fontsize":6},fmt ='0.2g', cmap="YlGnBu", xticklabels=x_label, yticklabels=y_label)
    ax.set_title('Period 4',fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)

    s = ax.get_figure()
    plt.subplots_adjust(hspace = 0.3)

    s.savefig('./spatial_correlation.pdf', dpi = 300, bbox_inches = 'tight')

def plot_in2():
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8,7),dpi=100)
    data = np.loadtxt('./in_2.txt')
    xaxis = ['Period 1','Period 2','Period 3','Period 4']
    plt.plot(xaxis, data[1], color='green', label='cam2',marker = 'o', markersize = 5,linestyle = 'dotted')
    plt.plot(xaxis, data[2], color='red', label='cam3',marker = 's', markersize = 5,linestyle = 'dotted')
    plt.plot(xaxis, data[3],  color='violet', label='cam4',marker = '+', markersize = 5,linestyle = 'dotted')
    plt.plot(xaxis, data[4], color='blue', label='cam5',marker = 'x', markersize = 5,linestyle = 'dotted')
    plt.plot(xaxis, data[5], color='gold', label='cam6',marker = 'd', markersize = 5,linestyle = 'dotted')
    plt.xticks(fontproperties = 'Times New Roman',fontsize=20)
    plt.yticks(fontproperties = 'Times New Roman',fontsize=20)
    plt.legend(fontsize=20,loc='upper left')
    plt.xlabel("Time Period",fontsize=30)
    plt.ylabel("Number of Visits",fontsize=30)
    plt.savefig('in_2.pdf')

    


def mat2mat(length):
    spa_cor = []

    for k in range(6):
        # model_path = './cam' + str(k+1) + '.mat'
        model_path = './cam' + str(k+1) + '_' + str(length) + '.mat'
        result = scipy.io.loadmat(model_path)
        spatial_distribution = result['rho']
        id = k
        res = []
        for j in range(length):
            tmp = []
            for i in range(6):
                if i == id:
                    tmp.append(0)
                
                tmp.append(spatial_distribution[0][i][j])
            tmp = tmp/sum(tmp)
            res.append(tmp)
        spa_cor.append(res)
    sol = []

    for i in range(length):
        tmp = spa_cor[0][i]
        for j in range(1,6):
            tmp = np.vstack((tmp,spa_cor[j][i]))
    
        
        sol.append(tmp)
    model_path = './correlation_predict_'+ str(length)+ '.mat'
    print(np.shape(sol))
    # print(sol[63])
    scipy.io.savemat(model_path, {'spatial':sol})



def resplot():
    plt.rcParams['pdf.fonttype'] = 42
    x  = [2,4,8,16]
    # y1 = [3.18,2.61,4.08,4.24]
    # y2 = [2.88,4.56,4.12,5.03]
    # y3 = [4.24,3.72,3.4,3.5]
    # y1 = [11,0,27,27]
    # y2 = [0,27,27,62]
    # y3 = [19,25,25,25]
    # plt.plot(x, y1, color = "red", label = 'static method')
    # plt.plot(x, y2, color = "blue", label = 'dynamic method')
    # plt.plot(x, y3, color = "green", label = 'predict method')
    # plt.legend()
    # plt.xlabel("time granularity")
    # plt.ylabel("miss number")
    # plt.savefig('res_3.png')
    # plt.ion()  # 开启交互模式
    # plt.subplots()

    # for j in range(20):
    #     plt.clf()     # 清空画布
    #     plt.xlim(0, 10)      # 因为清空了画布，所以要重新设置坐标轴的范围
    #     plt.ylim(0, 10)
        
    #     x = [random.randint(1,9) for i in range(10)]
    #     y = [random.randint(1,9) for i in range(10)]
        
    #     plt.scatter(x, y)
    #     plt.pause(0.2)
        
    #     plt.ioff()
    #     plt.show()

    N = 5
    #x1 = ['global','predict','dynamic','baseline','static']
    x1 = ['Global','Dynamic','MFG','Linear','Static']
    # x1 = ['Global','N$_t$ = 8','N$_t$ = 16','N$_t$ = 24','N$_t$ = 32']
    x = ['','','','','']
    y = [1343345,953909,896273,935888,937799]
    # plt.legend(loc='best')
    ax1 = plt.subplot(1, 3, 1)
    # plt.bar(range(N), y, color = 'lightsteelblue',label=x1)
    # plt.xticks(range(N), x)
    # plt.xlabel('Method')
    # plt.ylabel("Cost(frames)")
    # plt.legend()
    fig1_cost =[1343345,893272,896273,935888,937799]
    # fig1_cost = [1333800,34278,33045,85710,87516]
    # fig1_cost = [1343345,878946,910180,942400,945450]
    # fig1_cost =[1343345,942400,947105,880577,876138]
    a = ['Cost (k frames)']
    b_1 = fig1_cost[0]/1000
    b_2 = fig1_cost[1]/1000
    b_3 = fig1_cost[2]/1000
    b_4 = fig1_cost[3]/1000
    b_5 = fig1_cost[4]/1000

    bar_width = 0.1
    x_1 = list(range(len(a)))
    x_2 = [i+bar_width for i in x_1]
    x_3 = [i+bar_width*2 for i in x_1]
    x_4 = [i+bar_width*3 for i in x_1]
    x_5 = [i+bar_width*4 for i in x_1]

    # plt.figure(figsize=(20,8),dpi=80)
    plt.bar(range(len(a)),b_1,width=bar_width,label=x1[0],hatch= '/', color = '#FFC996')
    plt.bar(x_2,b_2,width=bar_width,label=x1[1],hatch = '+',fill = True, color = '#FFAB73')
    plt.bar(x_3,b_3,width=bar_width,label=x1[2],hatch  = 'X',fill = True, color = '#CA8A8B')
    plt.bar(x_4,b_4,width=bar_width,label=x1[3],fill = True, color = '#9F5F80')
    plt.bar(x_5,b_5,width=bar_width,label=x1[4],hatch = '\ ',fill = True, color = '#583D72')
    # plt.legend(loc='upper right')
    # plt.legend(loc='lower center' ,bbox_to_anchor=(0.5, -0.1))
    # plt.ylabel("Cost(frames)")
    fig1_recall = [0.7173333333333334,0.4573333333333333,0.424,0.43066666666666664,0.42533333333333334]
    # fig1_recall = [0.417910447761194,0.07164179104477612 ,0.03880597014925373,0.09850746268656717,0.09850746268656717]
    # fig1_recall = [0.7173333333333334,0.45066666666666666,0.464,0.476,0.4746666666666667]
    # fig1_recall = [0.7173333333333334,0.4666666666666667,0.424,0.38533333333333336,0.38666666666666666]    
    plt.xticks(x_3,a)
    ax2 = plt.subplot(1, 3, 2)
    a = ['Recall(%)']
    b_1 = fig1_recall[0]*100
    b_2 = fig1_recall[1]*100
    b_3 = fig1_recall[2]*100
    b_4 = fig1_recall[3]*100
    b_5 = fig1_recall[4]*100

    bar_width = 0.1
    x_1 = list(range(len(a)))
    x_2 = [i+bar_width for i in x_1]
    x_3 = [i+bar_width*2 for i in x_1]
    x_4 = [i+bar_width*3 for i in x_1]
    x_5 = [i+bar_width*4 for i in x_1]

    # plt.figure(figsize=(20,8),dpi=80)
    plt.bar(range(len(a)),b_1,width=bar_width,label=x1[0],hatch= '/', color = '#FFC996')
    plt.bar(x_2,b_2,width=bar_width,label=x1[1],hatch = '+',fill = True, color = '#FFAB73')
    plt.bar(x_3,b_3,width=bar_width,label=x1[2],hatch  = 'X',fill = True, color = '#CA8A8B')
    plt.bar(x_4,b_4,width=bar_width,label=x1[3],fill = True, color = '#9F5F80')
    plt.bar(x_5,b_5,width=bar_width,label=x1[4],hatch = '\ ',fill = True, color = '#583D72')
    # plt.legend(loc='best')
    # plt.legend(loc='upper right')
    # # plt.ylabel("Recall(%)")
    plt.legend(loc='lower center' ,ncol=5,bbox_to_anchor=(0.44, -0.16), prop={'size': 10}, handlelength = 3,handleheight = 1.5)

    plt.xticks(x_3,a)
    ax3 = plt.subplot(1, 3, 3)
    a = ['Precision(%)']
    fig1_precision = [0.8472440944881889,0.8448275862068966,0.828125,0.8197969543147208,0.8096446700507615]
    
    # fig1_precision = [0.6278026905829597,0.5581395348837209,0.6842105263157895,0.532258064516129,0.532258064516129]
    # fig1_precision = [0.8472440944881889,0.8666666666666667,0.8426150121065376,0.8188073394495413,0.8165137614678899]
    # fig1_precision = [0.8472440944881889,0.8027522935779816,0.7794117647058824,0.8525073746312685,0.8504398826979472]    
    b_1 = fig1_precision[0]*100
    b_2 = fig1_precision[1]*100
    b_3 = fig1_precision[2]*100
    b_4 = fig1_precision[3]*100
    b_5 = fig1_precision[4]*100

    bar_width = 0.1
    x_1 = list(range(len(a)))
    x_2 = [i+bar_width for i in x_1]
    x_3 = [i+bar_width*2 for i in x_1]
    x_4 = [i+bar_width*3 for i in x_1]
    x_5 = [i+bar_width*4 for i in x_1]

    # plt.figure(figsize=(20,8),dpi=80)
    plt.bar(range(len(a)),b_1,width=bar_width,label=x1[0],hatch= '/', color = '#FFC996')
    plt.bar(x_2,b_2,width=bar_width,label=x1[1],hatch = '+',fill = True, color = '#FFAB73')
    plt.bar(x_3,b_3,width=bar_width,label=x1[2],hatch  = 'X',fill = True, color = '#CA8A8B')
    plt.bar(x_4,b_4,width=bar_width,label=x1[3],fill = True, color = '#9F5F80')
    plt.bar(x_5,b_5,width=bar_width,label=x1[4],hatch = '\ ',fill = True, color = '#583D72')

    # plt.ylabel("Accuracy(%)")
    plt.xticks(x_3,a)
    




    plt.savefig('compare_method.pdf')


def plot_potential():
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8,7),dpi=100)
    xaxis = [1,2,3,4,6,8,10,12,14,16]
    data  = [3.54,5.19,11.92,19.41,21.85,37.4,46.1,49.04,69.27,67.57]
    plt.plot(xaxis, data, label='cam1',marker = 'o', markersize = 7,linestyle = 'dotted')
    plt.xticks(fontproperties = 'Times New Roman',fontsize=20)
    plt.yticks(fontproperties = 'Times New Roman',fontsize=20)
    # plt.legend(fontsize=10,loc='upper left')
    plt.xlabel("Time Granularity N$_t$",fontsize=29)
    plt.ylabel("Cost Ratio",fontsize=30)
    plt.savefig('save_ratio.pdf') 

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

def plot_range():
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8,6),dpi=300)
    plt.ylim((0, 1.2))
    xaxis = ['4','6','8','10','12','14']
    recall  = [0.4746666666666667/0.4826666666666667,1,0.464/0.4666666666666667,0.45066666666666666/0.448,0.444/0.44,0.4573333333333333/0.4226666666666667]
    precision = [0.8165137614678899/0.8302752293577982,1,0.8426150121065376/0.8027522935779816,0.8666666666666667/0.7924528301886793,0.8473282442748091/0.7764705882352941,0.8448275862068966/0.782716049382716]
    cost = [945450/951035,942400/954646,910180/953909,878946/952220,886129/953321,893272/946830]
    plt.plot(xaxis, recall, label='Recall',marker = 'o', markersize = 6,linestyle = '--',linewidth = 2, color = '#FFC996')
    plt.plot(xaxis, precision, label='Precision',marker = 'x', markersize = 6,linestyle = '--',linewidth = 2, color = 'red')
    plt.plot(xaxis, cost, label='Cost',marker = 'd', markersize = 6,linestyle = '--',linewidth = 2, color = '#583D72')

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
    plt.xticks(fontproperties = 'Times New Roman',fontsize=12)
    plt.yticks(fontproperties = 'Times New Roman',fontsize=12)
    plt.legend(fontsize=15,loc='lower right')
    plt.xlabel("Time Granularity (N$_t$)",fontsize=15)
    plt.ylabel("Value of Dyanmic Approach / Value of MFG",fontsize=15)
    plt.savefig('range.pdf') 



def main():
    # len = [36,40,44,48,52,56,60,64]
    # for l in len:
    #     mat2mat(l)
    # mat2mat(32)
    # dataPath = "../Market/pytorch/cameras/c1"
    # print(os.listdir(dataPath))
    plot_cdf()
    # y = np.loadtxt('global_24')
    # # print(a)
    # x = np.linspace(0,2500,50)
    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.bar(x=x, height=y)
    # ax.set_title("Simple Bar Plot", fontsize=15)
    # plt.savefig('cdf.png')
    # model_path = './correlation_split14_13.mat'
    # result = scipy.io.loadmat(model_path)
    # print(result['spatial'])  
    # model_path = './correlation_predict_64.mat'
    # result = scipy.io.loadmat(model_path)
    # spatial_distribution = result['spatial']
    # # tmp = spatial_distribution[]
    # # flag  = False
    # # for k in range(36, 40):

    # #     if not flag:
    # #         tmp_spatial = [[tmp[i][j]/(40/8) + spatial_distribution[k][i][j]/(40/8)  for j in range(len(tmp[0]))] for i in range(len(tmp))]
    # #         flag = True
    # #     else:
    # #         tmp_spatial = [[tmp[i][j] + spatial_distribution[k][i][j]/(40/8)  for j in range(len(tmp[0]))] for i in range(len(tmp))]
    # #     tmp = tmp_spatial
    # print(spatial_distribution[63])



if __name__ == '__main__':
    main()