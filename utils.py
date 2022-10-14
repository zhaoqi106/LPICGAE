import numpy as np
import torch


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def load_data(cuda):
    rnafeature = np.loadtxt('lnc_doc2vec_feature.csv', delimiter=',', encoding='UTF-8-sig')
    profeature = np.loadtxt('pro_doc2vec_feature.csv', delimiter=',', encoding='UTF-8-sig')
    lpi = np.loadtxt('adj_total.txt')

    lpi_torch = torch.from_numpy(lpi).float()
    rnafeature_torch = torch.from_numpy(rnafeature).float()
    profeature_torch = torch.from_numpy(profeature).float()

    graph_lnc = norm_adj(rnafeature)
    graph_pro = norm_adj(profeature)
    if cuda:
        rnafeature_torch = rnafeature_torch.cuda()
        profeature_torch = profeature_torch.cuda()
        lpi_torch = lpi_torch.cuda()
        graph_lnc = graph_lnc.cuda()
        graph_pro = graph_pro.cuda()
    return rnafeature_torch, profeature_torch, lpi_torch, graph_lnc, graph_pro


def norm_adj(feat):
    C = neighborhood(feat.T, k=10)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g


def neighborhood(feat, k):
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C


def normalized(wmat):
    deg = np.diag(np.sum(wmat, axis=0))
    degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


def metrix_convert(A):
    for i in range(len(A)):
        if A[i] >= 0.5:
            A[i] = 1
        else:
            A[i] = 0
    return A


def scaley(y):
    return (y - y.min()) / y.max()
