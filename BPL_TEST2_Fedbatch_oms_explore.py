# Setup application functions BPL_TEST2_Fedbatch_oms, dependent on previous import of functions 
# from fmu_explore
# Author: Jan Peter Axelsson
# -------------------------------------------------------------------------------------------------
# 2026-10-01 - Created
# 2026-10-01 - Import for oms relatted handling
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
#  Framework
# -------------------------------------------------------------------------------------------------

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------
#  Specific application functions: newplot(), describe()
# -------------------------------------------------------------------------------------------------


def newplot(title="Fedbatch cultivation", plotType="TimeSeries"):
    """Standard plot window
     title = ''
    two possible diagrams
     diagram = 'TimeSeries' default
     diagram = 'PhasePlane'"""

    resetPen()

    # Plot diagram
    if plotType == "TimeSeries":

        ax1 = plt.subplot(4, 1, 1)
        ax2 = plt.subplot(4, 1, 2)
        ax3 = plt.subplot(4, 1, 3)
        ax4 = plt.subplot(4, 1, 4)

        ax.clear()
        ax.append(ax1)
        ax.append(ax2)
        ax.append(ax3)
        ax.append(ax4)

        ax[0].set_title(title)
        ax[0].grid()
        ax[0].set_ylabel("X and S [g/L]")

        ax[1].grid()
        ax[1].set_ylabel("mu [1/h]")

        ax[2].grid()
        ax[2].set_ylabel("F [L/h]")

        ax[3].grid()
        ax[3].set_ylabel("V [L]")
        ax[3].set_xlabel("Time [h]")

        # List of commands to be executed by simu() after a simulation
        diagrams.clear()
        diagrams.append(
            "ax[0].plot(t,sim_res['bioreactor.c[1]'],color='r',linestyle=linetype)"
        )
        diagrams.append(
            "ax[0].plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)"
        )
        diagrams.append(
            "ax[0].legend(['X','S'])"
        )
        diagrams.append(
            "ax[1].plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)"
        )
        diagrams.append(
            "ax[2].plot(t,sim_res['bioreactor.inlet[1].F'],color='b',linestyle=linetype)"
        )
        diagrams.append(
            "ax[3].plot(t,sim_res['bioreactor.V'],color='b',linestyle=linetype)"
        )

    else:
        print("Plot window type not correct")

# -------------------------------------------------------------------------------------------------
#  Startup
# -------------------------------------------------------------------------------------------------

FMU_explore_info()
