import pyopencl as cl
import sys
sys.path.append('Q:\\python\\lib')
from utils import put, take
import threading
import numpy as np
from config import *
from rk import Rk

import time
from boundary import *
    
class Lg(threading.Thread):
    def __init__(self, B, T = None, X = None, V = None, update = None, maxiter = None, boundary = None, repeat = (0,None)):
        threading.Thread.__init__(self)
        self.repeat = repeat
        self.update = update
        self.maxiter = maxiter
        self.boundary = boundary
        self.shape = B.shape[:3]
        
        self.idx = np.indices((self.shape[0],self.shape[1]))
        self.idx[0] = self.idx[0] - self.shape[0]/2
        self.idx[1] = self.idx[1] - self.shape[1]/2
        
        
        self.step = np.float32(step)
        self.scale = np.float32(scale)
        self.clinit()
        self.loadData(B, T, X, V)
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
        
    def loadData(self, B, T, X, V):
        if (type(X) != np.ndarray):
            X = np.zeros(self.shape+(4,), dtype = np.float32)
            put(X,1.,3,3)
            
            
        if (type(V) != np.ndarray):
            V = np.zeros(self.shape+(4,), dtype = np.float32)
            put(V,1.,3,3)
            
        if (type(T) != np.ndarray):
            T = np.ones(self.shape, dtype = np.float32)
        
        self._X = X
        self._V = V
        self._T = T
        self._B = B
        
        self._slice = [np.zeros((self.shape[1], self.shape[2],4), dtype = np.float32),
                       np.zeros((self.shape[0], self.shape[2],4), dtype = np.float32),
                       np.zeros((self.shape[0], self.shape[1],4), dtype = np.float32)]
        
        self._F = np.zeros_like(V)
        put(self._F,1.,3,3)

        mf = cl.mem_flags
        self.size = X.nbytes
        
        self.X = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=X)
        self.Xtemp = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=X)
        self.Xnew = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=X)   
        
        self.V = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=V)
        self.Vtemp = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=V)
        self.Vnew = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=V)
        self.F = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self._F)
        
        self.T = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=T)        
        self.B = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)

        self.slice = [cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._slice[i]) for i in range(0,3)]
        
        self.queue.finish()     
    
    
    def put(self, X, V, ind, dim):
        self._slice[dim][:] = np.float32(V)

        cl.enqueue_copy(self.queue, self.slice[dim], self._slice[dim])
        self.program.Put(self.queue, self.shape, block_shape, 
                         X, self.slice[dim], np.uint32(ind), np.ubyte(dim))
        self.queue.finish()  
    
    def take(self, X, ind, dim):
        self.program.Take(self.queue, self.shape, block_shape, 
                          X, self.slice[dim], np.uint32(ind), np.ubyte(dim))
        cl.enqueue_copy(self.queue, self._slice[dim], self.slice[dim])
        self.queue.finish()  
        return self._slice[dim]
    
    
    def bound(self, a, dim, types):
        id1 = [0,self.shape[dim]-1]
        id2 = [self.shape[dim]-2, 1]

        for i in range(0,2):
            type = types[i]

            if (type == 'zero'):
                self.put(a, 0, id1[i], dim)
            else:

                if (type == 'uniform'):
                    ii = id2[i]
                    inv = 1

                if (type == 'continuous'):
                    ii = id2[i-1]
                    inv = 1

                if (type == 'mirror'):
                    ii = id2[i-1]
                    inv = -1

                self.put(a, inv*self.take(a, ii, dim), id1[i], dim)
    
    
    
    def setBoundary(self, type):
        for i in range(0,3):
            self.bound(self.F, i, type[i])
                

    
    def integrate(self):
        for i in range(0,self.rk.n):  
            self.program.Increment(self.queue, self.shape, block_shape, 
                                   self.Xtemp, self.X, self.Vtemp, 
                                   self.Vtemp, self.V, self.F, np.float32(step*self.rk.a[i]))


            self.program.Step(self.queue, self.shape, block_shape, 
                              self.Xtemp, self.Vtemp, self.F, self.T, self.B)         
            
            if (self.boundary != None):
                self.setBoundary(type = self.boundary)
            else:
                self.setBoundary()

            if (self.update != None):
                self.update(self, self.time + self.step*self.rk.a[i])


            self.program.Increment(self.queue, self.shape, block_shape, 
                                   self.Xnew, self.Xnew, self.Vtemp, 
                                   self.Vnew, self.Vnew, self.F, np.float32(step*self.rk.b[i]))
  
        cl.enqueue_copy(self.queue, self.X, self.Xnew)
        cl.enqueue_copy(self.queue, self.V, self.Vnew)

        self.queue.finish()
        self.time += self.step
    
    
    def run(self):
        self.run_key = True
        
        if (self.maxiter != None):
            self.niter = 0
        
        self.time = np.float32(0)
        
        while (self.run_key):
            #tt = time.time()
            self.integrate()
            
            if (self.maxiter != None):
                self.niter += 1
                if (self.niter == self.maxiter):
                    self.stop()
            
            if (self.repeat[0] != 0):
                if (self.niter % self.repeat[0] == 0):
                    self.repeat[1](self)
                    
            
            #print time.time()-tt
            
    def stop(self):
        self.run_key = False        
