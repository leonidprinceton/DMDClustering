import numpy
import scipy.linalg
import pylab
import random
from sklearn import linear_model

numpy.random.seed(0)
smaple_dt = 0.1
plot_dt = 0.001
omega1 = 10
omega2 = 15
sigma = 0.01
d = 4
t0 = -1
t1 = 1

data_func = lambda t: numpy.cos(omega1*t)*(t<0) + 2*numpy.cos(omega2*t - numpy.pi/3)*(t>=0)
t = numpy.arange(t0,t1+smaple_dt,smaple_dt)
x = data_func(t)

t_plot = numpy.arange(t0,t1+plot_dt,plot_dt)
x_plot = data_func(t_plot)

x += numpy.random.normal(0,sigma,x.shape)

pylab.figure(figsize=(6,5))
pylab.subplot(221)
pylab.title("(a)")
pylab.xlabel("time")
pylab.scatter(t, x, color="black", marker="^", s=20)
pylab.plot(t_plot, x_plot, linestyle="dashed", linewidth=1, color="black")

data = numpy.array([x[i:i+d] for i in range(len(x)-d+1)])
X = data[:-1]
Y = data[1:]

ransac = linear_model.RANSACRegressor(residual_threshold=1e-1)
ransac.fit(X, Y)

t = t[:-d]
for use_ransac in range(2):
	if use_ransac:
		X = X[ransac.inlier_mask_]
		Y = Y[ransac.inlier_mask_]
		t = t[ransac.inlier_mask_]

	U,s,V = scipy.linalg.svd(X, full_matrices=False)
	cutoff = 4
	U = U[:,:cutoff]
	V = V[:cutoff]
	s = s[:cutoff]

	K = U.T.dot(Y).dot(V.T).dot(numpy.diag(1/s))
	v,wl,wr = scipy.linalg.eig(K, left=True, right=True)

	pylab.subplot(222)
	pylab.scatter(numpy.real(v), numpy.imag(v), color="black", marker="so"[use_ransac], facecolor="none", s=30)

	pylab.subplot(224)
	pylab.xlabel("time")
	pylab.title("(d)")

	modes = numpy.abs(U.dot(wr))
	pylab.scatter(t, modes[:,0], color="black", marker="so"[use_ransac], s=30)
	pylab.scatter(t, modes[:,2], color="black", marker="so"[use_ransac], s=30, facecolor="none")

	pylab.subplot(223)
	pylab.title("(c)")
	pylab.xlabel("delays")

	K = V.dot(Y.T).dot(U).dot(numpy.diag(1/s))
	v,w = numpy.linalg.eig(K)
	modes = Y.T.dot(U).dot(numpy.diag(1/s)).dot(w)
	pylab.scatter(range(d), numpy.imag(modes[:,0]/modes[:,0][0]), color="black", marker="so"[use_ransac], s=30)
	pylab.scatter(range(d), numpy.imag(modes[:,2]/modes[:,2][0]), color="black", marker="so"[use_ransac], s=30, facecolor="none")

	t_delays = numpy.arange(0,d-1+plot_dt,plot_dt)
	pylab.plot(t_delays, numpy.sin(omega1*t_delays*smaple_dt), color="black", linestyle="dashed", linewidth=1)
	pylab.plot(t_delays, numpy.sin(omega2*t_delays*smaple_dt), color="black", linestyle="dashed", linewidth=1)

ax = pylab.subplot(222)
ax.add_artist(pylab.Circle((0,0), 1, facecolor="none", edgecolor="black", linestyle="dashed", linewidth=1))
pylab.xlim((-1.6,1.6))
pylab.ylim((-1.2,1.2))

eigvals = numpy.exp([complex(0,omega1*smaple_dt), complex(0,omega2*smaple_dt)])
eigvals = numpy.concatenate([eigvals,numpy.conj(eigvals)])
pylab.scatter(numpy.real(eigvals), numpy.imag(eigvals), marker="x", s=100, color="black")
pylab.title("(b)")
pylab.xlabel("real part")
pylab.ylabel("imaginary part")
pylab.tight_layout()
pylab.savefig("toy_example.pdf")
	
