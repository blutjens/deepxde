from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf
from numpy.random import default_rng

class Lorenz96Eq():
    def __init__(self, tgrid, plot=False):
        """
        Sets up Lorenz '96 equation as initial value problem. The equation
        models chaotic dynamics in three dimensions and is based on 
        Ch.2.1 in https://doi.org/10.1002/qj.2974 

        Args:
            tgrid np.array(n_grid): Grid points in time 

        Attributes:
            
        """
        self.tgrid = tgrid
        self.plot=plot
    
        seed = 1
        self.rng = default_rng(seed)

        self.K = 8 # Number of large-scale variables, x
        self.J = 8 # Number of middle-scale variables, y
        self.I = 8 # Number of small-scale variables, z
        self.h = 0.5 # Coupling strength between spatial scales; no coupling for h=0
        self.b = 10. # rel. magnitude of x, y
        self.c = 8. # rel. evolution speed of x, y
        self.d = 10. # rel. magnitude ratio of y, z
        self.e = 10. # rel. evolution speed of y, z
        self.F = 20. # Large-scale forcing

        self.hc_b = self.h * self.c / self.b # Coupling strength and ratio x, y
        self.he_d = self.h * self.e / self.d # Coupling strength and ratio y, z

        # Initial conditions
        self.x0 = np.random.randint(-5, 5, size=(self.K,))
        self.y0 = self.rng.standard_normal((self.J, self.K)) 
        self.z0 = .05 * self.rng.standard_normal((self.I, self.J, self.K))

        # Finite difference schemes in space; periodic boundaries
        if self.K != self.I or self.K != self.J:
            raise NotImplementedError("K, I, and J are assumed to be equal.")
        self.minus = np.roll(np.arange(0, self.K), 1) # e.g., [1, 2, 3, 0] for self.K = 4
        self.minus2 = np.roll(np.arange(0, self.K), 2)
        self.plus = np.roll(np.arange(0, self.K),-1)
        self.plus2 = np.roll(np.arange(0, self.K), -2)

        self.sample_random_params()

    def sample_random_params(self):
        """ Sample random parameters

        Sets all stochastic parameters to a new sample

        Returns:
        """
        return 1

    def step(self, x, y, z, verbose=False):
        """
        Compute 1st time derivative in tf2
        Source: Ch.2.1 in https://doi.org/10.1002/qj.2974 

        Args:
            x tf.Tensor(K,): Large-scale variables
            y tf.Tensor(J,K): Medium-scale variables
            z tf.Tensor(I,J,K): Small-scale variables
        Returns:
            dx_t tf.Tensor(K,): Derivative of large-scale variables wrt. time
            dy_t tf.Tensor(J,K): Derivative of medium-scale variables wrt. time
            dx_t tf.Tensor(I,J,K): Derivative of small-scale variables wrt. time
        """

        dx_t = tf.multiply(x[self.minus],(x[self.plus] - x[self.minus2])) - x + self.F - self.hc_b * tf.reduce_sum(y,axis=0)
        dy_t = - self.c * self.b * tf.multiply(y[self.plus,:], (y[self.plus2,:] - y[self.minus,:])) - self.c * y + self.hc_b * x - self.he_d * tf.reduce_sum(z,axis=0)
        dz_t = self.e * self.d * tf.multiply(z[self.minus,:,:],(z[self.plus,:,:] - z[self.minus2,:,:])) - self.e * z + self.he_d * y
        
        return dx_t, dy_t, dz_t

    def residual(self, x, y, z, t, verbose=False): # , b, c, d, e, h
        """
        Computes the residual in tf2
        Source: Ch.2.1 in https://doi.org/10.1002/qj.2974 

        Args:
            x tf.Tensor(K,): Large-scale variables
            y tf.Tensor(J,K): Medium-scale variables
            z tf.Tensor(I,J,K): Small-scale variables
            t tf.Tensor(n_tgrid,): Time vector
        Returns:
            [resx tf.Tensor(K,): Residual of large-scale variables 
            resy tf.Tensor(J,K): Residual of medium-scale variables
            resz tf.Tensor(I,J,K): Residual of small-scale variables
            ]
        """
        # Compute target time derivates with PDEs
        target_dx_t,target_dy_t,target_dz_t = step(x,y,t)
        
        # Compute estimated time derivatives with NN
        import pdb;pdb.set_trace()
        dx_t = dde.grad.jacobian(x, t)
        y = tf.Flatten(y)
        dy_t = dde.grad.jacobian(y, t)
        dy_t = dy_t.reshape(self.J,self.K)
        z = tf.Flatten(z)
        dz_t = dde.grad.jacobian(z, t) 
        dz_t = dz_t.reshape(self.I,self.J,self.K)
        
        # Compute residuals
        resx = dx_t - target_dx_t
        resy = dy_t - target_dy_t
        resz = dz_t - target_dz_t
        
        return [resx, resy, resz]

    def solve(self, euler=False):
        """
        Solves the equation

        Args:
            sol (np.array(n_tgrid, K),
                np.array(n_tgrid, J, K),
                np.array(n_tgrid, I, J, K),) 
        Returns:
        """
        if euler:
            raise NotImplementedError('Euler method not implemented for Lorenz96')
        else:
            dt = self.tgrid[-1] - self.tgrid[-2]
            solx = np.zeros((self.tgrid.shape[0], self.K))
            soly = np.zeros((self.tgrid.shape[0], self.J, self.K))
            solz = np.zeros((self.tgrid.shape[0], self.I, self. J, self.K))
            solx[0,:] = self.x0
            soly[0,:,:] = self.y0
            solz[0,:,:,:] = self.z0
            sol = (self.x0, self.y0, self.z0)

            for i, t in enumerate(self.tgrid[1:]):
                if i==self.tgrid.shape[0]:
                    break

                nvars = len(sol)
                dsol1 = self.step(*sol)

                Rsol2 = [sol[k] + 0.5 * dt * dsol1[k] for k in range(nvars)]
                dsol2 = self.step(*Rsol2)
                
                Rsol3 = [sol[k] + 0.5 * dt * dsol2[k] for k in range(nvars)]
                dsol3 = self.step(*Rsol3)
                
                Rsol4 = [sol[k] + dt * dsol3[k] for k in range(nvars)] 
                dsol4 = self.step(*Rsol4)

                solx[i,:], soly[i,:,:], solz[i,:,:,:] = [sol[k] + dt / 6. * (dsol1[k] + 2 * dsol2[k] + 2 * dsol3[k] + dsol4[k]) for k in range(nvars)]
                sol = (solx[i,:], soly[i,:,:], solz[i,:,:,:])

                if i%100 == 0: print(i)
        
        if self.plot:
            plotting.plot_lorenz96(self.tgrid, solx, soly, solz, self.K)
        return sol

def main():
    """
    Test multi-scale Lorenz96 
    author: Björn Lütjens (lutjens@mit.edu)
    """
    #######################v
    ##########
    # TODO: start here and deal with dimensions of X Y and Z
    ################
    dt = 0.005
    tmax = 10.
    tgrid = np.linspace(0.,tmax, int(tmax/dt)) # TODO: remove this tgrid, set time steps same as timedomain

    lorenz96 = Lorenz96Eq(tgrid)

    # Define PDE
    def pde(x, y):
        print('in pde()')
        print('x shape', x.shape)
        print('y shape', y.shape)
        import pdb;pdb.set_trace()
        #y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]

        t = x
        (x,y,z) = y
        resx, resy, resz = lorenz96.residual(x, y, z, t)

        return [resx,resy,resz]

    def solution(_):
        return lorenz96.solve()

    # Todo: find out what the time step is
    # Define temporal grid
    dt = 0.005
    tmax = 10.
    tgrid = np.linspace(0.,tmax, int(tmax/dt)) # TODO: remove this tgrid, set time steps same as timedomain
    geom = dde.geometry.TimeDomain(0., tmax)

    def boundary(_, on_initial):
        """TODO: Find out wth this fn is called boundary 
        """
        return on_initial

    # Initial conditions, lorenz
    icx = dde.IC(geom, lambda X: lorenz96.x0[:], boundary) # component indeces 
    icy = dde.IC(geom, lambda X: lorenz96.y0[:,:], boundary)#, component=0)
    icz = dde.IC(geom, lambda X: lorenz96.z0[:,:,:], boundary)#, component=1)

    # TODO: find out if I should load training data from file or define in solution() 
    # Get the train data 
    # observe_t, ob_y = gen_traindata()
    # observe_y0 = dde.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
    # observe_y1 = dde.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
    # observe_y2 = dde.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

    # bc = dde.DirichletBC(geomtime, solution, lambda _, on_boundary: on_boundary)
    data = dde.data.PDE(
        geom,
        pde,
        [icx, icy, icz],
        num_domain=400, # is this #time-steps?
        num_boundary=2,
        solution=solution,
    )

    layer_size = [lorenz.K] + [32] * 3 + [lorenz.J*lorenz.K]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)

    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()

