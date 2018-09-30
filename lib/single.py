import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from solver2 import Solver

class SingleSolver(Solver):
    def single_system(self, Y, t):
        v, hT = Y

        dv_dt = 1/self.C*(self.I_inj(t)*10**-3 - \
                          self.I_Kleak(v)*10**-2 - \
                          self.I_Naleak(v)*10**-2 -
                          self.I_T_no_time(v, hT)*10**-5)
        
        return dv_dt, self.dhT_dt(v, hT)

if __name__ == "__main__":
    t = sp.linspace(0, 10000, 10000)
    solver = SingleSolver()
    solver.step_amplitude = 0
    s = odeint(
        solver.single_system,
        [solver.Vrest, solver.hTrest], t)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, s[:,0])
    plt.show()
