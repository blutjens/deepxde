from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf
from numpy.random import default_rng
from deepxde.utils_folder import plotting

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

        self.K = 4 # Number of large-scale variables, x
        self.J = 4 # Number of middle-scale variables, y
        self.I = 4 # Number of small-scale variables, z
        self.h = 0.5 # Coupling strength between spatial scales; no coupling for h=0
        self.b = 10. # rel. magnitude of x, y
        self.c = 8. # rel. evolution speed of x, y
        self.d = 10. # rel. magnitude ratio of y, z
        self.e = 10. # rel. evolution speed of y, z
        self.F = 20. # Large-scale forcing

        self.xshape = (self.K,)
        self.yshape = (self.J, self.K)
        self.zshape = (self.I, self.J, self.K)
        self.shapes = (self.xshape, self.yshape, self.zshape)
        self.nscales = len(self.shapes) # number of scales

        self.hc_b = self.h * self.c / self.b # Coupling strength and ratio x, y
        self.he_d = self.h * self.e / self.d # Coupling strength and ratio y, z

        # Initial conditions
        self.x0 = np.random.randint(-5, 5, size=self.shapes[0]).astype(np.float32)
        self.y0 = self.rng.standard_normal(self.shapes[1], dtype=np.float32)
        self.z0 = .05 * self.rng.standard_normal(self.shapes[2], dtype=np.float32)
        self.sol0 = (tf.convert_to_tensor(self.x0), tf.convert_to_tensor(self.y0), tf.convert_to_tensor(self.z0))
        
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

    def step(self, x, y, z, parametrization='resolved', verbose=False):
        """
        Compute 1st time derivative in tf2
        Source: Ch.2.1 in https://doi.org/10.1002/qj.2974 

        Args:
            x tf.Tensor(K,): Large-scale variables; if input is of shape tf.Tensor(batch_size, K) only the first iteration will be used
            y tf.Tensor(J,K): Medium-scale variables
            z tf.Tensor(I,J,K): Small-scale variables
            parametrization str: Choice of parametrization: null, polynomial, superparam, resolved

        Returns:
            dx_t tf.Tensor(K,): Derivative of large-scale variables wrt. time
            dy_t tf.Tensor(J,K): Derivative of medium-scale variables wrt. time
            dx_t tf.Tensor(I,J,K): Derivative of small-scale variables wrt. time
        """
        # Convert batch into single input 
        # TODO: make function work for batch_size input
        if len(x.shape) > 1:
            x = x[0,:]
            if y is not None:
                y = y[0,:,:]
            if z is not None:
                z = z[0,:,:,:]

        # Calculate parametrization terms that captures influence of higher- to lower-resolution dynamics 
        if parametrization == 'null':
            x_param = tf.zeros_like(x)
        elif parametrization == 'resolved':
            x_param = - self.hc_b * tf.reduce_sum(y,axis=0)
            y_param = - self.he_d * tf.reduce_sum(z,axis=0)
        else:
            raise NotImplementedError("Parametrization " + parametrization + " not implemented.")

        # Calculate dynamics
        if verbose:
            import pdb;pdb.set_trace()
        dx_t = tf.multiply(tf.gather(x,self.minus),(tf.gather(x,self.plus) - tf.gather(x,self.minus2))) - x + self.F + x_param
        if parametrization == 'null':
            dy_t = tf.zeros((self.J, self.K))
            dz_t = tf.zeros((self.I, self.J, self.K))
        elif parametrization == 'resolved':
            dy_t = - self.c * self.b * tf.multiply(tf.gather(y,self.plus,0), (tf.gather(y,self.plus2,0) - tf.gather(y,self.minus,0))) - self.c * y + self.hc_b * x + y_param
            dz_t = self.e * self.d * tf.multiply(tf.gather(z,self.minus,0),(tf.gather(z,self.plus,0) - tf.gather(z,self.minus2,0))) - self.e * z + self.he_d * y
        return dx_t, dy_t, dz_t

    def residualx(self, x, t, parametrization = 'null', verbose=False):
        """
        Computes the residual on just the large-scale variables

        Args:
            x tf.Tensor(K,): Large-scale variables
            t tf.Tensor(n_tgrid,): Time vector
            parametrization str: Choice of parametrization: null, polynomial, superparam, resolved

        Returns:
            resx [tf.Tensor(K,)]: Residual of large-scale variables             
        """
        # Compute target time derivates with PDEs
        # TODO: Do I need to normalize target_dx_t with dt?
        target_dx_t, target_dy_t, target_dz_t = self.step(x, y=None, z=None, parametrization=parametrization, verbose=True)

        import pdb; pdb.set_trace()
        # Compute predicted time derivatives
        ##################################
        # TODO: start here: Why does dde.grad.jacobian return shape(1) instead of shape(self.K) tensor?
        ##################################
        dx_t = dde.grad.jacobian(x, t)        
        resx = dx_t - target_dx_t

        return [resx]

    def residual(self, x, y, z, t, parametrization='resolved', verbose=False): # , b, c, d, e, h
        """
        Computes the residual in tf2
        Source: Ch.2.1 in https://doi.org/10.1002/qj.2974 

        Args:
            x tf.Tensor(K,): Large-scale variables
            y tf.Tensor(J,K): Medium-scale variables
            z tf.Tensor(I,J,K): Small-scale variables
            t tf.Tensor(n_tgrid,): Time vector
            parametrization str: Choice of parametrization: null, polynomial, superparam, resolved

        Returns:
            [resx tf.Tensor(K,): Residual of large-scale variables 
            resy tf.Tensor(J,K): Residual of medium-scale variables
            resz tf.Tensor(I,J,K): Residual of small-scale variables
            ]
        """
        # Compute target time derivates with PDEs
        target_dx_t,target_dy_t,target_dz_t = self.step(x,y,t, parametrization=parametrization)
        
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
        Source: https://github.com/ashesh6810/Data-driven-super-parametrization-with-deep-learning

        Args:            
        Returns:
            sol [
            solx tf.Tensor(n_tgrid, K): Low-resolution variables
            soly tf.Tensor(n_tgrid, J, K): Medium-resolution variables
            solz tf.Tensor(n_tgrid, I, J, K)]: High-resolution variables
            ]: List of solutions
        """

        if euler:
            raise NotImplementedError('Euler method not implemented for Lorenz96')
        else:
            dt = self.tgrid[-1] - self.tgrid[-2]

            # Initialize solution
            sol = self.nscales*[]
            for k in range(self.nscales): 
                sol.append(tf.TensorArray(element_shape=self.shapes[k], size=self.tgrid.shape[0], dtype=tf.dtypes.float32, clear_after_read=False))
                sol[k] = sol[k].write(0,self.sol0[k])

            # Calculate solution forward in time
            for i, t in enumerate(self.tgrid[1:]):
                # TODO: document type of solver. 
                if i==self.tgrid.shape[0]:
                    break

                dsol1 = self.step(*[s.read(i) for s in sol])

                Rsol2 = [sol[k].read(i) + 0.5 * dt * dsol1[k] for k in range(self.nscales)]
                dsol2 = self.step(*Rsol2)
                
                Rsol3 = [sol[k].read(i) + 0.5 * dt * dsol2[k] for k in range(self.nscales)]
                dsol3 = self.step(*Rsol3)
                
                Rsol4 = [sol[k].read(i) + dt * dsol3[k] for k in range(self.nscales)] 
                dsol4 = self.step(*Rsol4)

                for k in range(self.nscales):
                    sol[k] = sol[k].write(i+1, sol[k].read(i) + dt / 6. * (dsol1[k] + 2 * dsol2[k] + 2 * dsol3[k] + dsol4[k]))

                if i%50 == 0: print('time-step: ', i)
    
        sol = [sol[k].stack() for k in range(self.nscales)]

        if self.plot:
            with tf.Session() as sess:
                plotting.plot_lorenz96(self.tgrid, sol[0].eval(), sol[1].eval(), sol[2].eval(), self.K)

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
    tf.disable_eager_execution() # Note: deepxde does not support eager

    # Define temporal grid
    dt = 0.005
    tmax = 0.25
    tgrid = np.linspace(0.,tmax, int(tmax/dt)) # TODO: remove this tgrid, set time steps same as timedomain
    # Todo: find out what the time step is
    geom = dde.geometry.TimeDomain(0., tmax)

    # Initialize ground-truth solver
    lorenz96 = Lorenz96Eq(tgrid, plot=True)

    # Define PDE
    def pde(x, y):# , X_in):
        """
        Compute the residual when learning NN: t -> [X_k]

        Args:
            x tf.Tensor(1,): independent variable, e.g., time, t
            y tf.Tensor(K,): Predicted variable, e.g., [X_k] or [Y_jk] 
            # X_in tf.Tensor(): Auxiliary variable, e.g., [X_k]
        Returns:
            res[
            resx tf.Tensor(K,): Residual of predicted large-scale variable
            ]: Residuals of each predicted variable
        """
        print('in pde()')

        t = x
        X_k = y
        print('t shape', t.shape)
        print('X_k shape', X_k.shape)
        res = lorenz96.residualx(X_k, t, parametrization='null')

        return res

    def solution(_):
        return lorenz96.solve()

    def boundary(_, on_initial):
        """TODO: Find out why this fn is called boundary 
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

    # bc = dde.sDirichletBC(geomtime, solution, lambda _, on_boundary: on_boundary)
    data = dde.data.PDE(
        geom,
        pde,
        [icx, icy, icz],
        num_domain=400, # is this #time-steps?
        num_boundary=2,
        solution=solution,
    )

    # layer_size = [lorenz96.K] + [32] * 3 + [lorenz96.J*lorenz96.K] # NN: t, [X_k] -> [Y_jk]
    layer_size = [1] + [32] * 1 + [lorenz96.K] # NN: t -> [X_k]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)

    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()

