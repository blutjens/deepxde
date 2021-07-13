import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm

"""
Plot Lorenz96
"""
def plot_lorenz96(tgrid, solx, soly, solz, K):
    """
    Plots the three solutions of lorenz96 
    """
    # Plot ensemble
    fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300)

    for k in range(K):
        if k>3:
            break
        axs[0].plot(tgrid, solx[:,k], label=r'$X_'+str(k)+'$')
        axs[1].plot(tgrid, soly[:,k,0], label=r'$Y_{'+str(k)+',0}$')
        axs[2].plot(tgrid, solz[:,k,0,0], label=r'$Z_{'+str(k)+',0,0}$')

    axs[0].set_ylabel(r'$X_t$')
    axs[0].set_xticks([])
    axs[1].set_ylabel(r'$Y_t$')
    axs[1].set_xticks([])
    axs[2].set_ylabel(r'$Z_t$')
    axs[2].set_xlabel(r'time, $t$')
    for i in range(3):    
        axs[i].legend(loc=1, prop={'size':6})

    axs[0].set_title("Lorenz '96")
    fig.tight_layout()
    fig.savefig('docs/figures/lorenz96_ens.png')
    plt.clf()

    # Plot single sample
    fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300)

    k = 0
    axs[0].plot(tgrid, solx[:,k], label=r'$X_'+str(k)+'$')
    axs[1].plot(tgrid, soly[:,k,k], label=r'$Y_{'+str(k)+','+str(k)+'}$')
    axs[2].plot(tgrid, solz[:,k,k,k], label=r'$Z_{'+str(k)+','+str(k)+','+str(k)+'}$')

    axs[0].set_ylabel(r'$X_t$')
    axs[0].set_xticks([])
    axs[1].set_ylabel(r'$Y_t$')
    axs[1].set_xticks([])
    axs[2].set_ylabel(r'$Z_t$')
    axs[2].set_xlabel(r'time, $t$')
    for i in range(3):    
        axs[i].legend(loc=1, prop={'size':6})

    axs[0].set_title("Lorenz '96")
    fig.tight_layout()
    fig.savefig('docs/figures/lorenz96.png')
    plt.clf()


