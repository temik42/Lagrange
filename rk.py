class Rk():
    def __init__(self, type = 'rk4'):
        if (type == 'rk4'):
            self.n = 4
            self.a = [0,0.5,0.5,1.]
            self.b = [1/6.,1/3.,1/3.,1/6.]