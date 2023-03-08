# source : https://github.com/P-N-Suganthan/CEC2017-BoundContrained/blob/master/Definitions%20of%20%20CEC2017%20benchmark%20suite%20final%20version%20updated.pdf

import numpy as np

class Ackley:   ## verified
    name = "Ackley"
    separable = True

    def __init__(self, d, a=20, b=0.2, c=2 * np.pi):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])
        self.a = a
        self.b = b
        self.c = c

    def get_param(self):
        return {"a": self.a, "b": self.b, "c": self.c}

    def get_global_minimum(self):
        X = np.array([0 for _ in range(d)])
        return (X, self(X))

    def __call__(self, X):
        res = -self.a * np.exp(-self.b * np.sqrt(np.mean(X**2)))
        res = res - np.exp(np.mean(np.cos(self.c * X))) + self.a + np.exp(1)
        return res

class Rosenbrock: # verified
    name = "Rosenbrock"
    separable = False

    def __init__(self, d, a=1, b=100):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])
        self.a = a
        self.b = b

    def get_param(self):
        return {"a": self.a, "b": self.b}

    def get_global_minimum(self):
        X = np.array([1 for _ in range(self.d)])
        return (X, self(X))

    def __call__(self, X):
        return np.sum(self.b * (X[1:] - X[:-1] ** 2) ** 2 + (self.a - X[:-1]) ** 2)

class Rastrigin: # verified
    name = "Rastrigin"
    separable = True

    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self):
        X = np.array([0 for _ in range(self.d)])
        return (X, self(X))

    def __call__(self, X): return 10 * self.d + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))

class PermZeroDBeta: # verified
    name = "Perm 0, d, beta"
    separable = False

    def __init__(self, d, beta=10):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])
        self.beta = beta

    def get_param(self):
        return {"beta": self.beta}

    def get_global_minimum(self):
        X = np.array([1 / (i + 1) for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X):
        j = np.arange(1, self.d + 1)
        res = np.sum(
            [
                np.sum((j + self.beta) * (X**i - 1/j**i)) ** 2 for i in range(1, self.d + 1)
            ]
        )
        return res

class Zakharov: ### Verified
    name = 'Zakharov'
    separable = False

    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self):
        X = np.array([0 for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X):
        i = np.arange(1, self.d + 1)
        return np.sum(X**2) + np.sum(0.5 * i * X) ** 2 + np.sum(0.5 * i * X) ** 4
    
class Schwefel: # verified
    name = 'Schwefel'
    separable = True
    
    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([420.9687 for i in range(self.d)])
        return (X, self(X))
    
    def __call__(self, X): return 418.9829*self.d - np.sum(X*np.sin(np.sqrt(np.abs(X))))
    

class Modified_Schwefel: #### Verified
    '''same as Schwefel function when evaluating on [-100,100]^d'''
    
    name = 'Modified Schwefel'
    separable = True
    
    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([0 for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X):
        def g(z):
            if np.abs(z) <= 500:
                return z*np.sin(np.sqrt(np.abs(z)))
            elif z > 500:
                return (500-z%500) * np.sin(np.sqrt(np.abs(500-z%500))) - (z-500)**2/(10000*self.d)
            elif z < -500:
                return (z%500-500) * np.sin(np.sqrt(np.abs(z%500-500))) - (z+500)**2/(10000*self.d)       
        Z = X + 420.9687462275036       
        return 418.9829*self.d - np.sum([g(z) for z in Z])
    
# source : http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
class Bent_Cigar: # verified
    name = 'Bent Cigar'
    separable = True
    
    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([420.968746 for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X): return X[0]**2 * 10**6 * np.sum(X[1:]**2)
    
    
class Expanded_Schaffer_f6: # verified
    name = 'Expanded Scaffer f6'
    separable = True
    
    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([0 for i in range(self.d)])
        return (X, self(X))
    
    def __call__(self, x):
        x_next = np.roll(x, -1)
        tmp = x ** 2 + x_next ** 2
        val = 0.5 + (np.sin(np.sqrt(tmp)) ** 2 - 0.5) / (1 + 0.001 * tmp) ** 2
        return np.sum(val)

class Levy: # verified
    name = 'Levy'
    separable = False
    
    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([1 for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X):
        W = 1 + (X-1)/4
        return np.sin(np.pi*W[0])**2 + np.sum((W[:-1]-1)**2 * (1+10*np.sin(np.pi*W[:-1]+1)**2)) + (W[-1]-1)**2 * (1+np.sin(2*np.pi*W[-1])**2)
    
class High_Conditioned_Elliptic: # verified
    name = 'High Conditioned Elliptic'
    separable = True
    
    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([420.968746 for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X): return np.sum((10**6) ** (np.arange(self.d) / (self.d - 1)) * X**2)

class Discus:  # verified
    name = 'Discus'
    separable = True
    
    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([420.968746 for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X): return 10**6 * X[0]**2 + np.sum(X[1:]**2)
    
class Weierstrass:  ################### VERIFIED
    name = 'Weierstrass'
    separable = False
    
    def __init__(self, d, a=0.5, b=3, kmax=20):
        self.d = d
        self.a = a
        self.b = b
        self.kmax = kmax
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {'a': self.a, 'b': self.b, 'kmax': self.kmax}
    
    def get_global_minimum(self):
        X = np.array([420.968746 for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X):   
        return np.sum(np.sum([self.a**k * np.cos(2*np.pi*self.b**k*(X+0.5))  for k in range(self.kmax+1)])) - self.d * np.sum([self.a**k * np.cos(2*np.pi*0.5) for k in range(self.kmax+1)])
    

class Griewank: # verified
    name = 'Griewank'
    separable = False
    
    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([0 for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X): return np.sum(X**2)/4000 - np.prod([np.cos(X[i]/np.sqrt(i+1)+1) for i in range(self.d)])

# source : https://niapy.org/en/stable/_modules/niapy/problems/katsuura.html
class Katsuura: # verified
    name = 'Katsuura'
    separable = False
    
    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([420.968746 for i in range(self.d)])
        return (X, self(X))

    def __call__(self, X):
        
        val = 1.0
        for i in range(self.d):
            val_t = 1.0
            for j in range(1, 33):
                val_t += np.abs(2 ** j * X[i] - round(2 ** j * X[i])) / 2 ** j
            val *= (1 + (i + 1) * val_t) ** (10 / self.d ** 1.2) - (10 / self.d ** 2)
        return 10 / self.d ** 2 * val
    
class Happy_Cat: # verified
    name = 'Happy Cat'
    separable = False
    
    def __init__(self, d, alpha=0.25):
        self.d = d
        self.alpha = alpha
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {'alpha': self.alpha}
    
    def get_global_minimum(self):
        X = np.array([420.968746 for i in range(self.d)])
        return (X, self(X))
    
    def __call__(self, X): return np.abs(np.sum(X**2 - self.d))**self.alpha + 0.5*np.sum((X**2 + X))/self.d + 0.5
    
class HGBat:
    name = 'HGBat'
    separable = False
    
    def __init__(self, d, alpha=0.25):
        self.d = d
        self.alpha = alpha
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {'alpha': self.alpha}
    
    def get_global_minimum(self):
        X = np.array([420.968746 for i in range(self.d)])
        return (X, self(X))
    
    def __call__(self, X): return np.abs(np.sum(X**2)**2 - np.sum(X)**2)**0.5 + 0.5*np.sum((X**2 + X))/self.d + 0.5


class Lunacek_bi_Rastrigin: 
    name = 'Lunacek-bi-Rastrigin'
    separable = False
    
    def __init__(self, d, a=0.5, b=2.0, h=1.0):
        self.d = d
        self.a = a
        self.b = b
        self.h = h
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {'a': self.a, 'b': self.b, 'd': self.d}
    
    def get_global_minimum(self):
        X = np.array([0 for i in range(self.d)]) # UNKNOWN
        return (X, self(X))
    
    def __call__(self, X):
        s1 = np.sum((X-self.a)**2 - 10*np.cos(np.pi*(X-self.a)))
        s2 = np.sum((X-self.a)**2 - 10*np.cos(np.pi*(X+self.a)))
        return self.h*np.abs(s1-s2) + self.b*np.sqrt(self.d) + 10*self.d
    

class expanded_griewank_plus_rosenbrock:
    name = 'expanded Griewank plus Rosenbrock'
    separable = False
    
    def __init__(self, d, a=0.5, b=2.0, h=1.0):
        self.d = d
        self.input_domain = np.array([[-100, 100] for _ in range(d)])

    def get_param(self):
        return {}
    
    def get_global_minimum(self):
        X = np.array([0 for i in range(self.d)]) # UNKNOWN
        return (X, self(X))
    
    def __call__(self, X):
        f_7 = Lunacek_bi_Rastrigin(1)
        f_4 = Rosenbrock(2)
        return np.sum([f_7(f_4(np.array([X[i],X[i+1]]))) for i in range(self.d-2)]) + f_7(f_4(np.array([X[-1],X[0]])))

    
    
    
    
    
    
    
    
    
    
