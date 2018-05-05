from tkinter import filedialog
import pandas as pd
from tkinter import *
import numpy as np
from sklearn.decomposition import PCA
from numpy import *
from sklearn import svm
window=Tk()
window.title("SVM")
#window.geometry('500x300')

#h函数
#主成分
def index_lst(lst, component=0, rate=0):
    if component and rate:
        print('Component and rate must choose only one!')
        sys.exit(0)
    if not component and not rate:
        print('Invalid parameter for numbers of components!')
        sys.exit(0)
    elif component:
        print('Choosing by component, components are %s......' % component)
        return component
    else:
        print('Choosing by rate, rate is %s ......' % rate)
        for i in range(1, len(lst)):
            if sum(lst[:i]) / sum(lst) >= rate:
                return i
        return 0


def main(x):
    # test data
    mat =x

    # simple transform of test data
    Mat = np.array(mat, dtype='float64')
    print('Before PCA transforMation, data is:\n', Mat)
    print('\nMethod 1: PCA by original algorithm:')
    p, n = np.shape(Mat)  # shape of Mat
    t = np.mean(Mat, 0)  # mean of each column

    # substract the mean of each column
    for i in range(p):
        for j in range(n):
            Mat[i, j] = float(Mat[i, j] - t[j])

    # covariance Matrix
    cov_Mat = np.dot(Mat.T, Mat) / (p - 1)

    # PCA by original algorithm
    # eigvalues and eigenvectors of covariance Matrix with eigvalues descending
    U, V = np.linalg.eigh(cov_Mat)
    # Rearrange the eigenvectors and eigenvalues
    U = U[::-1]
    for i in range(n):
        V[i, :] = V[i, :][::-1]
    # choose eigenvalue by component or rate, not both of them euqal to 0
    Index = index_lst(U, component=2)  # choose how many main factors
    if Index:
        v = V[:, :Index]  # subset of Unitary matrix
    else:  # improper rate choice may return Index=0
        print('Invalid rate choice.\nPlease adjust the rate.')
        print('Rate distribute follows:')
        print([sum(U[:i]) / sum(U) for i in range(1, len(U) + 1)])
        sys.exit(0)
    # data transformation
    T1 = np.dot(Mat, v)
    #return T1
    # print the transformed data
    #print('We choose %d main factors.' % Index)
    print('After PCA transformation, data becomes:\n', T1)
    return T1
#读取文件路径
def opentrain():
    global fn
    train1 = filedialog.askopenfilename(title='打开校正集', filetypes=[('train', '*.xls'), ('All Files', '*')])
    fn=train1
    return(train1)

def opentest():
    global fm
    test1 = filedialog.askopenfilename(title='打开验证集/待测数据', filetypes=[('test', '*.xls'), ('All Files', '*')])
    fm=test1
    return test1

def opentrain1():
    global fe
    train2 = filedialog.askopenfilename(title='打开校正集化学值', filetypes=[('化学值', '*.xls'), ('All Files', '*')])
    fe=train2
    return train2

def opentest1():
    global fr
    test2 = filedialog.askopenfilename(title='打开验证集化学值', filetypes=[('验证集化学值', '*.xls'), ('All Files', '*')])
    fr=test2
    return test2

#SVM算法
def svm_1():
    global fn,fm,fe,fr,fw
    a = np.array(pd.read_excel(fn, header=None))
    b = np.array(pd.read_excel(fe, header=None))
    c = np.array(pd.read_excel(fm, header=None))
    #d = np.array(pd.read_excel(fr, header=None))
    x = np.array(list(map(lambda a: a[1:], a))).T
    y = b.ravel()
    #d = d.ravel()
    c = np.array(list(map(lambda x: x[1:], c))).T
    x = main(x)
    c = main(c)

    C1=eval(a1.get())
    gamma1=eval(b1.get())
    clf = svm.SVC(kernel='rbf', random_state=0, gamma=gamma1, C=C1)
    clf.fit(x, y)
    fw=clf.predict(c)
    d1.set(fw)
    #print("SVM-输出测试集的准确率为：", clf.score(c, d))
    #e1.set(clf.score(c,d))
#查看准确率
def check():
    global fw
    print(fw)
    K=0
    global fr
    if fr==0:
        e1.set("无对比值")
    else:
        d = np.array(pd.read_excel(fr, header=None))
        d = d.ravel()
        if(len(fw)!=len(d)):
            e1.set("请更新对比值")
        else:
            for i in range(0,len(d)):
                if d[i]==fw[i]:
                    K+=1
            e1.set(K/len(d))





#调参
Label(window,text="调参准备").grid(row=1,column=0,columnspan=2)

Label(window,text="C:").grid(row=2,column=0,sticky=E)

Label(window,text="gamma:").grid(row=2,column=2,sticky=E)

Label(window,text="预测结果:",font=('宋体', 15, 'bold')).grid(row=4,column=0,sticky=E)

#输入参数
#参数C
a1=StringVar()
entenNum1=Entry(window,width=8,textvariable=a1)
entenNum1.grid(row=2,column=1,sticky=W)
#
b1=StringVar()
entenNum2=Entry(window,width=18,textvariable=b1)
entenNum2.grid(row=2,column=3,sticky=W)

d1=StringVar()
entenNum4=Entry(window,width=25,state="readonly",textvariable=d1)
entenNum4.grid(row=4,column=1,columnspan=3,sticky=NSEW)

e1=StringVar()
entenNum4=Entry(window,width=25,state="readonly",textvariable=e1)
entenNum4.grid(row=5,column=2,columnspan=2,sticky=W)
#按钮

btn1 = Button(window, text='打开校正集', font=('宋体', 15, 'bold'), command=opentrain).grid(padx=3,pady=5,column=0, row=0)

btn2 = Button(window, text='打开校正集化学值', font=('宋体', 15, 'bold'), command=opentrain1).grid(padx=3,pady=5,column=1, row=0)

btn3 = Button(window, text='打开验证/待测数据', font=('宋体', 15, 'bold'), command=opentest).grid(padx=3,pady=5,column=2, row=0)

btn4 = Button(window, text='打开验证集化学值', font=('宋体', 15, 'bold'), command=opentest1).grid(padx=3,pady=5,column=3, row=0)

btn5=Button(window,text="SVM",font=('宋体', 15, 'bold'), command=svm_1).grid(row=3,column=0,columnspan=4,padx=15)

btn6 = Button(window, text='查看准确率', font=('宋体', 15, 'bold'),command=check).grid(pady=5,column=1, row=5)
#具体运行
global fn,fm,fe,fw
fr=0
mainloop()
#使用
