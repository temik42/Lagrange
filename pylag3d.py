import pyopencl as cl
import sys
import threading
import numpy as np
from config import *
from rk import Rk

#a = [0,0.5,0.5,1.]
#b = [1/6.,1/3.,1/3.,1/6.]    

    
class Lg(threading.Thread):
    def __init__(self, B, X = None, V = None, update = None, maxiter = None, boundary = None, repeat = (0,None)):
        threading.Thread.__init__(self)
        self.repeat = repeat
        self.update = update
        self.maxiter = maxiter
        self.boundary = boundary
        self.shape = B.shape[0:3]
        self.step = np.float32(step)
        self.scale = np.float32(scale)
        self.clinit()
        self.loadData(X, V, B)
        self.loadProgram("Q:\\python\\Lagrange\\pylag3d.cl")
        self.rk = Rk()
        #self.a = a
        #self.b = b
        
    def clinit(self):
        plats = cl.get_platforms()
        
        if glEnable:
            from pyopencl.tools import get_gl_sharing_context_properties
            if sys.platform == "darwin":
                self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                                 devices=[])
            else:
                self.ctx = cl.Context(properties=[
                    (cl.context_properties.PLATFORM, plats[0])]
                    + get_gl_sharing_context_properties(), devices=None)
                
        else:
            self.ctx = cl.create_some_context()
            
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        kernel_params = {"block_size": block_size, "scale": self.scale, "nx": self.shape[0], "ny": self.shape[1], "nz": self.shape[2]}
        self.program = cl.Program(self.ctx, fstr % kernel_params).build()
        
    def loadData(self, X, V, B):
        if (X == None):
            X = np.zeros_like(B)
            X[:,:,:,3] = 1.
            
        if (V == None):
            V = np.zeros_like(B)
            V[:,:,:,3] = 1.
        
        self._X = X
        self._V = V
        self._X1 = np.array(X)
        self._V1 = np.array(V)
        self._B = B
        self._Fx = np.zeros_like(X)
        self._Fv = np.zeros_like(V)
        #self._Bx = np.array(B)
        #self._Error = np.zeros(self.shape, dtype = np.float32)
        #self._Current = np.zeros(self.shape+(4,), dtype = np.float32)
        mf = cl.mem_flags
        self.size = X.nbytes
        
        self.X = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=X)
        self.Fx = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self._Fx)
        self.V = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=V)
        self.Fv = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self._Fv)
        self.B = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        #self.Bx = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        self.DB = cl.Buffer(self.ctx, mf.READ_WRITE, self.size*3)  
        #self.Current = cl.Buffer(self.ctx, mf.READ_WRITE, self.size)
        #self.Error = cl.Buffer(self.ctx, mf.READ_WRITE, self.size/4)

        self.queue.finish()     
    
    def setBoundary(self, type = ['continuous','continuous','continuous']):

        
        self._Fx[0,:,:,:] = -self._Fx[-2,::-1,:,:]
        self._Fx[-1,:,:,:] = -self._Fx[1,::-1,:,:]
        self._Fx[:,0,:,:] = -self._Fx[::-1,-2,:,:]
        self._Fx[:,-1,:,:] = -self._Fx[::-1,1,:,:]
        self._Fx[0,:,:,2] = self._Fx[-2,::-1,:,2]
        self._Fx[-1,:,:,2] = self._Fx[1,::-1,:,2]
        self._Fx[:,0,:,2] = self._Fx[::-1,-2,:,2]
        self._Fx[:,-1,:,2] = self._Fx[::-1,1,:,2]
        
        
        if (type[0] == 'uniform'):
            self._Fx[0,:,:,:] = self._Fx[-2,:,:,:]
            self._Fx[-1,:,:,:] = self._Fx[1,:,:,:]
        if (type[1] == 'uniform'):    
            self._Fx[:,0,:,:] = self._Fx[:,-2,:,:]
            self._Fx[:,-1,:,:] = self._Fx[:,1,:,:]
        if (type[2] == 'uniform'):
            self._Fx[:,:,0,:] = self._Fx[:,:,-2,:]
            self._Fx[:,:,-1,:] = self._Fx[:,:,1,:]
            
        if (type[0] == 'continuous'):
            self._Fx[0,:,:,:] = self._Fx[1,:,:,:]
            self._Fx[-1,:,:,:] = self._Fx[-2,:,:,:]
        if (type[1] == 'continuous'):    
            self._Fx[:,0,:,:] = self._Fx[:,1,:,:]
            self._Fx[:,-1,:,:] = self._Fx[:,-2,:,:]
        if (type[2] == 'continuous'):
            self._Fx[:,:,0,:] = self._Fx[:,:,1,:]
            self._Fx[:,:,-1,:] = self._Fx[:,:,-2,:]
            
        if (type[0] == 'mirror'):
            self._Fx[0,:,:,:] = -self._Fx[1,:,:,:]
            self._Fx[-1,:,:,:] = self._Fx[-2,:,:,:]
        if (type[1] == 'mirror'):    
            self._Fx[:,0,:,:] = -self._Fx[:,1,:,:]
            self._Fx[:,-1,:,:] = -self._Fx[:,-2,:,:]
        if (type[2] == 'mirror'):
            self._Fx[:,:,0,:] = -self._Fx[:,:,1,:]
            self._Fx[:,:,-1,:] = -self._Fx[:,:,-2,:]
        
        if (type[0] == 'fixed'):
            self._Fx[0,:,:,:] = 0
            self._Fx[-1,:,:,:] = 0
        if (type[1] == 'fixed'):    
            self._Fx[:,0,:,:] = 0
            self._Fx[:,-1,:,:] = 0
        if (type[2] == 'fixed'):
            self._Fx[:,:,0,:] = 0
            self._Fx[:,:,-1,:] = 0
            

    
    def integrate(self,a,b):
        if (self.boundary != None):
            self.setBoundary(type = self.boundary)
        else:
            self.setBoundary()
        if (self.update != None):
            self.update(self, self.time + self.step*a)
        cl.enqueue_copy(self.queue, self.X, self._X + self._Fx*step*a)
        cl.enqueue_copy(self.queue, self.V, self._V + self._Fv*step*a)
        cl.enqueue_barrier(self.queue)
        self.program.Step(self.queue, self.shape, block_shape, 
                                   self.X, self.Fx, self.V, self.Fv, self.B, self.DB, 
                                   np.float32(self.step))
        cl.enqueue_barrier(self.queue)
        cl.enqueue_read_buffer(self.queue, self.Fx, self._Fx)
        cl.enqueue_read_buffer(self.queue, self.Fv, self._Fv)
        cl.enqueue_barrier(self.queue)
        if (self.boundary != None):
            self.setBoundary(type = self.boundary)
        else:
            self.setBoundary()
        if (self.update != None):
            self.update(self, self.time + self.step*a)

        self._X1 += self._Fx*step*b
        self._V1 += self._Fv*step*b
        self.queue.finish()
    
    
    def run(self):
        self.program.Jacobian(self.queue, self.shape, block_shape, self.B, self.DB)
        self.queue.finish()
        self.run_key = True
        
        if (self.maxiter != None):
            self.niter = 0
        
        self.time = np.float32(0)
        
        while (self.run_key):
            for i in range(0,self.rk.n):
                self.integrate(self.rk.a[i],self.rk.b[i])
            
            self._X = np.array(self._X1, dtype = np.float32)
            self._V = np.array(self._V1, dtype = np.float32)

            self.time += self.step
            
            self.queue.finish()
            if (self.maxiter != None):
                self.niter += 1
                if (self.niter == self.maxiter):
                    self.stop()
            
            if (self.repeat[0] != 0):
                if (self.niter % self.repeat[0] == 0):
                    self.repeat[1](self)
            
            
    def stop(self):
        self.run_key = False        
