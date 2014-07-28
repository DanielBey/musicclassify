import numpy as np
from collections import defaultdict

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit

from sklearn.metrics import confusion_matrix

from utils import plot_roc, plot_confusion_matrix, GENRE_LIST

from ceps import read_ceps

TEST_DIR = "测试集的文件夹"

genre_list = GENRE_LIST#体裁分类列表

#clf_factory是逻辑回归模型
#X中是600行13列的一个矩阵。代表600首歌，每首歌有13个特征
#Y中是对应的这个歌曲属于哪一类只有600行
#name是"Log Reg CEPS"  plot控制是否做图
def train_model(clf_factory, X, Y, name, plot=False):
    labels = np.unique(Y)#得到分类列表
    #随机地从X的600个元素中选出30%作为测试集，选1次
    cv = ShuffleSplit(
        n=len(X), n_iter=1, test_size=0.3, indices=True, random_state=0)

    train_errors = []
    test_errors = []

    scores = []#用于保存测试集的准确率
    pr_scores = defaultdict(list)
    precisions, recalls, thresholds = defaultdict(
        list), defaultdict(list), defaultdict(list)

    roc_scores = defaultdict(list)
    tprs = defaultdict(list)#假正率字典
    fprs = defaultdict(list)#真正率字典

    clfs = []  # just to later get the median

    cms = []

    for train, test in cv:
        #train中是420个600内的随机数test是另外180个数
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]
        #现在X_train中存着训练集的420个13维向量y_train存着他们对应的420个类名
        #X_test中存着测试集的180个13维向量y_test存着他们对应的180个类名
        clf = clf_factory()
        clf.fit(X_train, y_train)#利用训练集训练出逻辑回归模型
        
        clfs.append(clf)#将每次训练集回归出的模型塞进去

        train_score = clf.score(X_train, y_train)#用训练出的模型检测训练集的准确率
        test_score = clf.score(X_test, y_test)#用训练出的模型检测测试集的准确率
        scores.append(test_score)

        train_errors.append(1 - train_score)#训练集的错误率
        test_errors.append(1 - test_score)#测试集的错误率

        y_pred = clf.predict(X_test)#预测测试集中的180首歌分别对应的类型
        cm = confusion_matrix(y_test, y_pred)#
        cms.append(cm)

        for label in labels:
            #y_test是180行代表每首歌真实属于哪个类别
            y_label_test = np.asarray(y_test == label, dtype=int)
            proba = clf.predict_proba(X_test)#X_test是180行13列
            #proba是180行6列每一列是这首歌曲被分为6个类别各自的概率
            proba_label = proba[:, label]
            #proba_label是180行1列是proba中的某一列
            #precision_recall_curve需要两个参数
            #y_label_test是01序列180行，代表每首歌是否是label类
            #proba_label也是个180行的序列，代表每首歌被预测为label类的概率
            precision, recall, pr_thresholds = precision_recall_curve(
                y_label_test, proba_label)
            pr_scores[label].append(auc(recall, precision))
            precisions[label].append(precision)
            recalls[label].append(recall)
            thresholds[label].append(pr_thresholds)

            fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
            roc_scores[label].append(auc(fpr, tpr))
            tprs[label].append(tpr)
            fprs[label].append(fpr)
    
    if plot:
        for label in labels:
            print "Plotting", genre_list[label]
            scores_to_sort = roc_scores[label]
            median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

            desc = "%s %s" % (name, genre_list[label])
            plot_roc(roc_scores[label][median], desc, tprs[label][median],
                     fprs[label][median], label='%s vs rest' % genre_list[label])

    all_pr_scores = np.asarray(pr_scores.values()).flatten()
    summary = (np.mean(scores), np.std(scores),
               np.mean(all_pr_scores), np.std(all_pr_scores))
    print "%.3f\t%.3f\t%.3f\t%.3f\t" % summary

    return np.mean(train_errors), np.mean(test_errors), np.asarray(cms)


def create_model():
    from sklearn.linear_model.logistic import LogisticRegression
    clf = LogisticRegression()

    return clf


if __name__ == "__main__":
    X, y = read_ceps(genre_list)
    #X中存储了所有种类的歌曲总数个13维向量。向量中每一维代表歌曲的一个特征。
    #y中是对应的这个歌曲属于哪一类
    train_avg, test_avg, cms = train_model(
        create_model, X, y, "Log Reg CEPS", plot=True)

    cm_avg = np.mean(cms, axis=0)
    cm_norm = cm_avg / np.sum(cm_avg, axis=0)

    print cm_norm

    plot_confusion_matrix(cm_norm, genre_list, "ceps",
                          "Confusion matrix of a CEPS based classifier")
