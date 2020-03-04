#!/usr/bin/env python
'''
SVM character recognition.

Sample loads a dataset of characters from 'charctest12ALL.png'.
Then it trains a SVM classifiers on it and evaluates
their accuracy.

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))

Influenced by design of
[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
'''

import numpy as np
import cv2
from multiprocessing.pool import ThreadPool
import itertools as it
from numpy.linalg import norm

SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 36 #number of characters to be trained
charc_FN = 'CharTrainImg.png'#dataset to be trained





def split2d(img, cell_size, flatten=True):
    '''split2d used to split the dataset vertically and
    horizontally and get all individual charachters'''
    h, w = img.shape[:2]#get shape of the dataset
    sx, sy = cell_size#size of individual charachters
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]#each individual charachters found
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells



def load_charc(fn):
    ''' load individual charachters and their respective labels
    from learning dataset'''
    print 'loading "%s" ...' % fn
    charc_img = cv2.imread(fn, 0)
    charc = split2d(charc_img, (SZ, SZ))#all charachters found
    labels = np.repeat(np.arange(CLASS_N), len(charc)/CLASS_N)
    return charc, labels#each digit and its label in terms of row number is returned

   

def deskew(img):
    '''deskew every character'''
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)




class SVM(StatModel):
    '''svm class definition with train predict and initialize functions'''
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()


def evaluate_model(model, charc, samples, labels):
    '''test the accuracy of the  model after training it'''
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print "resp:",resp
    print "labels",labels
    print 'error: %.2f %%' % (err*100)
    
    confusion = np.zeros((CLASS_N, CLASS_N), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print 'confusion matrix:'
    print confusion
    

    vis = []
    for img, flag in zip(charc, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    
    vis = iter(vis)
    img0 = vis.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], vis)
    args = [iter(vis)] * 25
    rows= it.izip_longest(fillvalue=pad, *args)
    return np.vstack(map(np.hstack, rows))
    

    return np.float32(charc).reshape(-1, SZ*SZ) / 255.0

def preprocess_hog(charc):
    '''find HOG of every charachter'''
    samples = []
    for img in charc:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)#x axis gradient
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)#y axis gradient
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16# 16 bin histogram
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]#divide each bin image into 4 parts
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]#divide each mag image into 4 parts
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]#obtain histogram
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


if __name__ == '__main__':
    print __doc__

    charc, labels = load_charc(charc_FN)
    

    print 'preprocessing...'
    # shuffle charc
    rand = np.random.RandomState(321)
    print len(charc)
    shuffle = rand.permutation(len(charc))
    charc, labels = charc[shuffle], labels[shuffle]

    charc2 = map(deskew, charc)
    samples = preprocess_hog(charc2)

    train_n = int(0.75*len(samples))
    charc1=charc[train_n:]
    charc1 = iter(charc1)
    img1 = charc1.next()
    pad = np.zeros_like(img1)
    imgs = it.chain([img1], charc1)
    args = [iter(charc1)] * 25
    rows= it.izip_longest(fillvalue=pad, *args)
    testset=np.vstack(map(np.hstack, rows))
    
    cv2.imshow('Test set', testset)
    charc_train, charc_test = np.split(charc2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])
    
    print 'training SVM...'
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, charc_test, samples_test, labels_test)
    cv2.imshow('SVM test', vis)
    print 'saving SVM as "charc_svm.dat"...'
    model.save('charc_svm.dat')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
