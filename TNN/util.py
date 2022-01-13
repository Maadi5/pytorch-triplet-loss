import numpy as np
from matplotlib import pyplot as plt
import random
import torch
from TNN import Mining

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


def split_label_class(labels, num_split):
	labels.pop(labels.index('N/A'))
	set1 = random.sample(labels, num_split)
	set2 = []
	for i in labels:
		if i not in set1:
			set2.append(i)
	return set1, set2


def get_lbl_val(classes_all, subset):
	s2 = []
	for i in subset:
		s2.append(classes_all.index(i))
	return s2


def find_index_elementlist(main_arr, id_list):
	index_lbl = []
	main_arr = np.array(main_arr)
	for ind, g in enumerate(id_list):
		ar = np.where(main_arr == g)
		ar_l = ar[0].tolist()
		index_lbl.extend(ar_l)
	return index_lbl

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
	
def compare_embeddings(input1, input2, thresh_val):
    concatenated = torch.cat((input1,input2),0).cuda()
    embeddings = model(concatenated)
    dists = Mining._pairwise_distances(embeddings, squared=True, device=device)
    dist = dists.cpu().detach()
    if float(dist[0][1])> thresh_val: #avg_intra_dist.avg:
      prediction = 0
    else:
      prediction = 1
    return prediction
