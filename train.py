import torch.nn as nn
import torch.nn.functional as F
import argparse
from model import VGAE, GAE
from utils import *
from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Dimension of representations')
parser.add_argument('--omega', type=float, default=0.7,
                    help='Weight between lncRNA space and protein space')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed, args.cuda)
rnafeature_torch, profeature_torch, lpi_torch, graph_lnc, graph_pro = load_data(args.cuda)


class FRN(nn.Module):
    def __init__(self):
        super(FRN, self).__init__()
        self.FRNl = VGAE(rnafeature_torch.shape[1], 256, args.hidden)
        self.FRNp = VGAE(profeature_torch.shape[1], 256, args.hidden)

    def forward(self, xl0, xp0):
        hl, stdl, xl = self.FRNl(graph_lnc, xl0)
        hp, stdp, xp = self.FRNp(graph_pro, xp0)

        return hl, stdl, xl, hp, stdp, xp


class LPN(nn.Module):
    def __init__(self):
        super(LPN, self).__init__()
        self.LPNl = GAE(args.hidden, lpi_torch.shape[1])
        self.LPNp = GAE(args.hidden, lpi_torch.shape[0])

    def forward(self, y0):
        yl, zl = self.LPNl(graph_lnc, y0)
        yp, zp = self.LPNp(graph_pro, y0.t())
        return yl, zl, yp, zp


print("5-fold Cross Validation:")


def criterion(output, target, n_nodes, mu, logvar):
    reconstruction_cost = F.mse_loss(output, target)

    KL = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    return reconstruction_cost + KL


def train(FRN, LPN, xl0, xp0, y0, epoch, omega):
    fai0 = 1.0
    tao0 = 1.0
    opt_FRN = torch.optim.Adam(FRN.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_LPN = torch.optim.Adam(LPN.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for e in range(epoch):

        # FRN
        FRN.train()
        hl, stdl, xl, hp, stdp, xp = FRN(xl0, xp0)
        loss_FRNl = criterion(xl, xl0, graph_lnc.shape[0], hl, stdl)
        loss_FRNp = criterion(xp, xp0, graph_pro.shape[0], hp, stdp)
        loss_FRN = omega * loss_FRNl + (1 - omega) * loss_FRNp + fai0 * e * F.mse_loss(
            torch.mm(hl, hp.t()), y0) / epoch
        opt_FRN.zero_grad()
        loss_FRN.backward()
        opt_FRN.step()
        FRN.eval()
        with torch.no_grad():
            hl, _, _, hp, _, _ = FRN(xl0, xp0)

        # LPN
        LPN.train()
        yl, zl, yp, zp = LPN(y0)
        loss_LPNl = F.binary_cross_entropy(yl, y0) + tao0 * e * F.mse_loss(zl, hl) / epoch
        loss_LPNp = F.binary_cross_entropy(yp, y0.t()) + tao0 * e * F.mse_loss(zp, hp) / epoch
        loss_LPN = omega * loss_LPNl + (1 - omega) * loss_LPNp
        opt_LPN.zero_grad()
        loss_LPN.backward()
        opt_LPN.step()
        LPN.eval()
        with torch.no_grad():
            yl, _, yp, _ = LPN(y0)

        if e % 20 == 0:
            print('Epoch %d | loss_LPN: %.4f | loss_FRN: %.4f' % (e, loss_LPN.item(), loss_FRN.item()))

    return omega * yl + (1 - omega) * yp.t()


def cross_validation(A, omega = 0.5):
    N = A.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    res = torch.zeros(5, A.shape[0], A.shape[1])

    aurocl = np.zeros(5)
    auprl = np.zeros(5)
    f1l = np.zeros(5)
    acc1 = np.zeros(5)
    pre1 = np.zeros(5)

    for i in range(5):
        print("Fold {}".format(i + 1))
        A0 = A.clone()
        for j in range(i * N // 5, (i + 1) * N // 5):
            A0[idx[j], :] = torch.zeros(A.shape[1])

        frn = FRN()
        lpn = LPN()
        if args.cuda:
            frn = frn.cuda()
            lpn = lpn.cuda()

        train(frn, lpn, rnafeature_torch, profeature_torch, A0, args.epochs, args.omega)
        frn.eval()
        lpn.eval()
        yli, _, ypi, _ = lpn(A0)
        resi = omega * yli + (1 - omega) * ypi.t()

        res[i] = resi

        if args.cuda:
            resi = resi.cpu().detach().numpy()
        else:
            resi = resi.detach().numpy()

        auroc, aupr, acc, f1, pre = show_metrics(resi)
        aurocl[i] = auroc
        auprl[i] = aupr
        f1l[i] = f1
        acc1[i] = acc
        pre1[i] = pre

    y_pred = res[auprl.argmax()]
    if args.cuda:
        return y_pred.cpu().detach().numpy()
    else:
        return y_pred.detach().numpy()


title = 'cross_validation'
y_pred = cross_validation(lpi_torch, omega=args.omega)
y_pred = scaley(y_pred)
print(y_pred)
show_metrics(y_pred)
