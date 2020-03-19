import pickle
from visdom import Visdom
import numpy as np
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def save_obj(obj, filename, verbose=True):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    if verbose:
        logger.info("Saved object to %s." % filename)


def load_obj(filename, verbose=True):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    if verbose:
        logger.info("Load object from %s." % filename)
    return obj


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def accuracy(df, out, label, index):

    qtypes = df['QType'].to_list()

    acc_total, acc_vis, acc_text, acc_tem, acc_know = 0, 0, 0, 0, 0
    num_vis, num_text, num_tem, num_know = 0, 0, 0, 0

    for o, l, i in zip(out, label, index):

        if o == l:
            acc_total += 1

        qtype = qtypes[i]

        if qtype == 'visual':
            num_vis += 1
            if o == l:
                acc_vis += 1
        elif qtype == 'textual':
            num_text += 1
            if o == l :
                acc_text += 1
        elif qtype == 'temporal':
            num_tem += 1
            if o == l:
                acc_tem += 1
        elif qtype == 'knowledge':
            num_know += 1
            if o == l:
                acc_know += 1

    acc_total = acc_total / len(out)
    acc_vis = acc_vis / num_vis
    acc_text = acc_text / num_text
    acc_tem = acc_tem / num_tem
    acc_know = acc_know / num_know

    logger.info('--- Accuracy')
    logger.info('Total: %.03f' % acc_total)
    logger.info('Visual : %.03f' % acc_vis)
    logger.info('Textual : %.03f' % acc_text)
    logger.info('Temporal : %.03f' % acc_tem)
    logger.info('Knowledge : %.03f' % acc_know)
    logger.info('------')

    return acc_total, acc_vis, acc_text, acc_tem, acc_know