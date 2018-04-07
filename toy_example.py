import numpy
import scipy.linalg
import pylab
import random
from sklearn import linear_model

pylab.rc("text", usetex=True)
pylab.rc("font", family="serif")
numpy.random.seed(1)

pylab.figure(figsize=(5,4))	
rand_sinusoid = lambda omega,length: (1+numpy.random.random())*numpy.real(numpy.exp(complex(0, numpy.random.random()*numpy.pi*2) + complex(0,omega)*numpy.arange(length)))
omegaA = 1.0
omegaB = 1.7
omegaC = 0.8
omegaD = 1.5
d = 19

ax = pylab.subplot(222)
ax.add_artist(pylab.Circle((0,0), 1, facecolor="none", edgecolor="black", linestyle="dashed", linewidth=1))
pylab.xlim((-1.6,1.6))
pylab.ylim((-1.2,1.2))

eigvals = numpy.exp([complex(0,omegaA), complex(0,omegaD), complex(0,omegaC), complex(0,omegaB)])
eigvals = numpy.concatenate([eigvals,numpy.conj(eigvals)])
pylab.scatter(numpy.real(eigvals), numpy.imag(eigvals), marker="x", s=100, color="black")
pylab.title("(b)")
pylab.ylabel(r"$Im\left\{ \lambda\right\} $")
pylab.xlabel(r"$Re\left\{ \lambda\right\} $")


data = numpy.array([rand_sinusoid(omegaA,d+1) for _ in range(6)] + [rand_sinusoid(omegaB,d+1) for _ in range(6)] + [rand_sinusoid(omegaC,d+1) + rand_sinusoid(omegaD,d+1) for _ in range(11)])
data = data + numpy.random.normal(0,0.1,data.shape)
X = data[:,:-1]
Y = data[:,1:]
t = numpy.arange(X.shape[0])

U,s,V = scipy.linalg.svd(X, full_matrices=False)


pylab.subplot(221)
pylab.scatter(range(len(s)), s, color="black", marker=".", facecolor="none", s=30)
pylab.ylim((1e-2,1e2))
pylab.title("(a)")
pylab.xlabel(r"$i$")
pylab.ylabel(r"$\sigma_{i}$")
pylab.yscale("log")

cutoff = 8
U = U[:,:cutoff]
V = V[:cutoff]
s = s[:cutoff]

K = U.T.dot(Y).dot(V.T).dot(numpy.diag(1/s))
v,wl,wr = scipy.linalg.eig(K, left=True, right=True)

modes = numpy.abs(U.dot(wr))

markers = "oo^^ssDD" + "*"*(len(v)-8)
colors = "ggrrbbyy" + "k"*(len(v)-8)
for i in range(8):
	pylab.subplot(222)
	pylab.scatter([v[i].real], [v[i].imag], color="black", marker=markers[i], facecolor=colors[i], s=30)

	if i%2 == 0:
		if i/2 in (1,2):
			pylab.subplot(223)
			pylab.scatter(t, modes[:,i], color="black", marker=markers[i], facecolor=colors[i], s=30)
		else:
			pylab.subplot(224)
			pylab.scatter(t, modes[:,i], color="black", marker=markers[i], facecolor=colors[i], s=30)


pylab.subplot(224)
pylab.title("(d)")
pylab.ylabel(r"$\left| q_{5,7}(i)\right|$")
pylab.xlabel(r"$i$")

pylab.subplot(223)
pylab.title("(c)")
pylab.ylabel(r"$\left| q_{1,3}(i)\right|$")
pylab.xlabel(r"$i$")

pylab.tight_layout()
pylab.savefig("toy_example.pdf")
#pylab.show()
	
