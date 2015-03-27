# A script for making a movie that shows the evolution of vorticity field
# Data is taken from KBTurbulence.txt

import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
from mayavi import mlab
from numpy import *
Ngrid = 512
Nsteps = 400

Nvar = 1
Nframes =Nsteps

s=mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(0, 0, 0),size = (700,700))
mlab.view(0, 0, 90, (0.0, 0.0, 0.10))
s.scene.camera.parallel_projection = 1
filew = open('KBTurbulence.txt','r')

data = loadtxt(filew,comments='P')
data = data.reshape(Nsteps,Ngrid,Ngrid)

for steps in xrange(Nframes-1):
    omega= data[steps,:,:]
    mlab.clf()
    mlab.contour_surf(omega/abs(omega.max()),colormap = 'jet', opacity =.95, line_width= 4.0,transparent = True,figure=s)
    im=mlab.surf(omega/abs(omega.max()),colormap = 'jet', opacity =.80, transparent = True,warp_scale='auto',figure=s)
    fname = 'KB_fig%03d.png'%steps
    mlab.savefig(fname)
os.system("convert -quality 100 *.png KBturbMovie.gif")
os.system("rm KB_fig*.png")
    
