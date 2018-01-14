import scipy.misc
import scipy.linalg
import pylab
import numpy
import sys
from math import *
import scipy.sparse.linalg
import sklearn.cluster
import sklearn.neighbors
import itertools
import sklearn.linear_model

Ny = 50
cutoff = 13
n_clusters = 6

img = scipy.misc.imread("TEM.jpg")[:,:,0]

x,y = numpy.meshgrid(range(img.shape[1]), range(img.shape[0]))
	
data = numpy.concatenate([[x[Ny/2:-Ny/2,Ny/2:-Ny/2].reshape((-1))], [y[Ny/2:-Ny/2,Ny/2:-Ny/2].reshape((-1))]], axis=0)
connectivity = sklearn.neighbors.kneighbors_graph(data.T, 8+16, include_self=False)


delayed_data = []
TLS = True

for i in range(Ny/2, img.shape[0]-Ny/2):
	for j in range(Ny/2, img.shape[1]-Ny/2):
		delayed_data.append(img[i,j-Ny/2:j+Ny/2])
		delayed_data.append(img[i-Ny/2:i+Ny/2,j])

last_shape = (img.shape[0] - Ny, img.shape[1] - Ny)
delayed_data = numpy.array(delayed_data).T

X = delayed_data[:-1].T
Y = delayed_data[1:].T

U,s,V = scipy.sparse.linalg.svds(X.astype(numpy.float32), k=cutoff)

X_ = U.T.dot(X)
Y_ = U.T.dot(Y)

if TLS:
	XX_ = numpy.concatenate([X_,Y_])
	U_,s_,V_ = scipy.linalg.svd(XX_, full_matrices=False)
	K = U_[cutoff:,:cutoff].dot(numpy.linalg.inv(U_[:cutoff,:cutoff]))
else:
	K = U.T.dot(Y).dot(V.T).dot(numpy.diag(1/s))

def join_pairs(Ms,w):
	Ms_ = [Ms[0]]
	w_ = [w[0]]
	for i in range(1, len(w)):
		if abs(numpy.conj(w[i]) - w_[-1]) < 1e-10:
			Ms_[-1] += numpy.conj(Ms[i])
		else:
			w_.append(w[i])
			Ms_.append(Ms[i])
	return numpy.array(Ms_), numpy.array(w_)

v,wl = scipy.linalg.eig(K, left=True, right=False)
wl,v = join_pairs(wl.T,v)
wl = wl.T

FF = numpy.abs(wl.T.dot(U.T))
FF = numpy.concatenate([FF[:,::2], FF[:,1::2]], axis=0)

pylab.figure(figsize=(8,2))
p = 9

pylab.subplot(121)
pylab.title("(a)")
pylab.imshow(FF[p].reshape(last_shape), cmap="gray", vmin=0, vmax=0.015)
pylab.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off") 
pylab.tick_params(axis="y", which="both", right="off", left="off", labelleft="off") 

pylab.subplot(122)
pylab.title("(b)")
pylab.imshow(FF[p%len(v)].reshape(last_shape), cmap="gray", vmin=0, vmax=0.015)
pylab.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off") 
pylab.tick_params(axis="y", which="both", right="off", left="off", labelleft="off") 
pylab.tight_layout()
pylab.savefig("TEM_modes.pdf")

pylab.figure(figsize=(9,3))

img_3c = numpy.zeros((img.shape[0], img.shape[1], 3))
img_3c[:,:,0] = img/255.0
img_3c[:,:,1] = img/255.0
img_3c[:,:,2] = img/255.0

y = 171
x = 287
pylab.subplot(121)
pylab.title("(a)")

img_3c[y-1:y+2,x-Ny/2:x+Ny/2,2] = 0
img_3c[y-1:y+2,x-Ny/2:x+Ny/2,1] = 0
img_3c[y-1:y+2,x-Ny/2:x+Ny/2,0] = 1
img_3c[y-Ny/2:y+Ny/2,x-1:x+2,0] = 0
img_3c[y-Ny/2:y+Ny/2,x-1:x+2,1] = 0
img_3c[y-Ny/2:y+Ny/2,x-1:x+2,2] = 1
pylab.imshow(img_3c)
pylab.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off") 
pylab.tick_params(axis="y", which="both", right="off", left="off", labelleft="off") 

pylab.subplot(122)
pylab.title("(b)")
pylab.plot(range(-Ny/2,Ny/2), img[y,x-Ny/2:x+Ny/2], color="red")
pylab.plot(range(-Ny/2,Ny/2), img[y-Ny/2:y+Ny/2,x], color="blue")
pylab.xlabel("pixel offset")
pylab.ylabel("brightness")

t = numpy.arange(Ny)
pylab.tight_layout()
pylab.savefig("TEM_scans.pdf")

pylab.figure(figsize=(6,3))
clustering = sklearn.cluster.AgglomerativeClustering(linkage="ward", n_clusters=n_clusters, connectivity=connectivity).fit(FF.T)

img_clusters = numpy.zeros((img.shape[0], img.shape[1], 3))
img_clusters[:,:,0] = img/255.0
img_clusters[:,:,1] = img/255.0
img_clusters[:,:,2] = img/255.0
mask = clustering.labels_.reshape((img.shape[0] - Ny, img.shape[1] - Ny))#/float(n_clusters)

colorz = [map(bool, (i&1,i&2,i&4)) for i in range(1,8)]

mask_r = sum((mask == i)*colorz[i][0] for i in range(n_clusters))
mask_g = sum((mask == i)*colorz[i][1] for i in range(n_clusters))
mask_b = sum((mask == i)*colorz[i][2] for i in range(n_clusters))

img_clusters[Ny/2:Ny/2+mask.shape[0],Ny/2:Ny/2+mask.shape[1],0] *= mask_r
img_clusters[Ny/2:Ny/2+mask.shape[0],Ny/2:Ny/2+mask.shape[1],1] *= mask_g
img_clusters[Ny/2:Ny/2+mask.shape[0],Ny/2:Ny/2+mask.shape[1],2] *= mask_b
pylab.imshow(img_clusters)
pylab.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off") 
pylab.tick_params(axis="y", which="both", right="off", left="off", labelleft="off") 

pylab.savefig("TEM_clusters.pdf")

