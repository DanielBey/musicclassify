import os
import glob
import sys

import numpy as np
import scipy
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

from utils import GENRE_DIR

#将求得的MFCC参数ceps放入文件*.ceps.npy中
#原文件的每个*.wav都对应一个*.ceps.npy
def write_ceps(ceps, fn):
    """
    Write the MFCC to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print "Written", data_fn


def create_ceps(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    #X包含的是所有采样的样本
    ceps, mspec, spec = mfcc(X)
    write_ceps(ceps, fn)

#genre_list是所有分类的列表
def read_ceps(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        #现在genre代表每一个类型
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            num_ceps = len(ceps)
            X.append(
                np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
                )#将4135帧中的前后10%去掉并对剩余的3300多行的13列分别取平均数
            #塞进X的是一个个1行13列的向量
            y.append(label)#塞入这个歌曲所属的类别

    return np.array(X), np.array(y)


if __name__ == "__main__":
    os.chdir(GENRE_DIR)#进入存放所有歌曲文件夹的总目录genres
    glob_wav = os.path.join(sys.argv[1], "*.wav")#类似于jazz/ *.wav
    print glob_wav
    for fn in glob.glob(glob_wav):#遍历jazz文件夹下的所有wav文件
        create_ceps(fn)
