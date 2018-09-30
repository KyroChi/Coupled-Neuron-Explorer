"""
Plots an interactive figure that includes the voltage of an
injected and a coupled neuron. The figure includes sliders
for the leak conductences, a plot of the voltages and 
injected currents and a bifurcation diagram bifurcating on
the potassium and sodium leak conductences.
"""

from lib.solver import Solver
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from matplotlib.gridspec import GridSpec, \
    GridSpecFromSubplotSpec
from matplotlib.widgets import Slider

class CoupledSolver(Solver):
    """
    Coupled version of the standard solver. Differential 
    Equation for a coupled system of neurons rather than a
    single one.
    """
    g_kleak2 = Solver.g_kleak
    g_naleak2 = Solver.g_naleak
    
    def I_Kleak (self, V, g_k):
        """ 
        Modified potassium leak channel that accepts 
        multiple different potassium channel leak 
        conductences, since we are modeling two differing
        channel conductences.
        """
        return g_k * (V - self.E_kleak)

    def I_Naleak (self, V, g_na):
        """
        Modified sodium leak channel that accepts multiple
        different sodium channel leak conductences, since we
        are modeling two differing channel conductences.
        """
        return g_na * (V - self.E_naleak)

    def I_leak (self, V, cell='injected'):
        """
        Combined leak channel
        """
        return (self.g_kleak + self.g_naleak)*(V - (
            self.g_kleak*self.E_kleak + \
            self.g_naleak*self.E_naleak)/(
                self.g_kleak + self.g_naleak)
        )


    def coupled_system(self, Y, t):
        """
        Defines a coupled system for the scipy integrator
        """
        V1, V2, hT1, hT2 = Y
        
        dV1_dt = (self.S/self.C*10**3) * \
                 (
                     self.I_inj(t)/self.S*10**-3 - \
                     self.p_T*hT1*self.G(V1)*\
                     self.m_Tinf(V1)**2*10**-3 - \
                     self.I_leak(V1, 'injected') - \
                     self.gbar_h*self.m_hinf(V1)*(V1-self.E_h) -
                     self.ggj*(V1 - V2)
                 )
        
        dV2_dt = (self.S/self.C*10**3) * \
                 (
                     self.ggj*(V1 - V2) - \
                     self.p_T*hT2*self.G(V2)*\
                     self.m_Tinf(V2)**2*10**-3 - \
                     self.gbar_h*self.m_hinf(V1)*(V1-self.E_h) -
                     self.I_leak(V2, 'injected')
                 )
        
        dhT1_dt = self.dhT_dt(V1, hT1)
        dhT2_dt = self.dhT_dt(V2, hT2)
        
        return dV1_dt, dV2_dt, dhT1_dt, dhT2_dt


    def injected_V_nullcline (self, V1, V2, injection=0):
        """ """
        return (
            ( injection/self.S*10**-3 - \
              self.I_leak(V1, 'injected') -\
              self.ggj*(V1 - V2)) /
            ( self.P_T(V1)*self.m_Tinf(V1)**2\
              *self.G(V1)*10**-3 )
        )
        

    def coupled_V_nullcline (self, V1, V2, injection=0):
        """
        """
        return (
            ( self.ggj*(V1 - V2) - \
              self.I_leak(V2, 'injected') ) /
            ( self.P_T(V2)*self.m_Tinf(V2)**2\
              *self.G(V2)*10**-3 )
        )

    def coupled_system_2(self, Y, t):
        """
        Defines a coupled system for the scipy integrator
        """
        V1, V2, hT1, hT2, mh1, mh2 = Y
        
        dV1_dt = (1/self.C) * \
                 (
                     self.I_inj(t)*10**-3 -
                     self.I_T_no_time(V1, hT1)*10**-5 -
                     self.I_h(V1, mh1)*10**-2 -
                     self.I_leak(V1)*10**-2 -
                     self.ggj*(V1 - V2)*self.S*10**-2
                 )
        
        dV2_dt = (1/self.C) * \
                 (
                     self.ggj*(V1 - V2)*self.S*10**-2 -
                     self.I_T_no_time(V2, hT2)*10**-5 -
                     self.I_h(V2, mh2)*10**-2 -
                     self.I_leak(V2)*10**-2
                 )
        
        dhT1_dt = self.dhT_dt(V1, hT1)
        dhT2_dt = self.dhT_dt(V2, hT2)

        dmh1_dt = self.dmh_dt(V1, mh1)
        dmh2_dt = self.dmh_dt(V2, mh2)
        
        return (dV1_dt, dV2_dt, dhT1_dt, dhT2_dt, dmh1_dt,
                dmh2_dt)

    def injected_V_nullcline_2 (self, V1, V2, injection=0):
        """ """
        return (
            ( injection*10**-3 - 
              self.I_leak(V1)*10**-2 -
              self.ggj*(V1 - V2)*self.S*10**-2) /
            ( self.P_T(V1)*self.m_Tinf(V1)**2\
              *self.G(V1) )
        )
        

    def coupled_V_nullcline_2 (self, V1, V2, injection=0):
        """
        """
        return (
            ( self.ggj*(V1 - V2)*self.S*10**-2 - 
              self.I_leak(V2)*10**-2 ) /
            ( self.P_T(V2)*self.m_Tinf(V2)**2\
              *self.G(V2) )
        )


def coupled_system_grid(total_time=10000,
                        fidelity=10000):
    """ 
    Build a grid interface for looking at the dynamics of a
    coupled system and its associated bifurcation steady
    state plot. Also defines the update callback for updating
    the display with the new dynamics of the system.

    Caller must call plt.show() to display the graph.

    @param total_time: The time over which the interface will
    be integrating over. Time is measured in ms. Defaults to 
    10,000 (100 seconds).
    @param fidelity: The number of samples over the time.
    Defaults to 10,000, or 1 step per ms.
    @return: None
    """
    s = CoupledSolver()
    s.step_type = 'drop'
    s.step_initial = -10
    s.c = 'mh'
    
    if s.c is 'mh':
        # Change the resting voltage in the mh scheme
        # TODO: Add this to the table?
        s.Vrest = -64
        
    inj_amp = -4
    inj_start = 0
    inj_dur = 3000
    
    t = sp.linspace(0, total_time, fidelity)
    fig = plt.figure(figsize=(15,8))
    gs_master = GridSpec(3, 2, height_ratios=[0.1, 0.3, 10],
                         wspace=0.10, hspace=0.25, top=0.95,
                         bottom=0.075)
    gs_1 = GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs_master[0, :])
    title_axes = fig.add_subplot(gs_1[0])
    title_axes.set_title(
        'Coupled IT Ih Model Neurons', fontsize=30)
    title_axes.axis('off')

    gs_2 = GridSpecFromSubplotSpec(
        1, 9, subplot_spec=gs_master[1, :])

    amp_slider = fig.add_subplot(gs_2[0])
    s_amp = Slider(amp_slider, 'Injection Amplitude', -100,
                   10, valinit=inj_amp)
    dur_slider = fig.add_subplot(gs_2[1])
    s_dur = Slider(dur_slider, 'Duration', 0, total_time,
                   valinit=0)
    str_slider = fig.add_subplot(gs_2[2])
    s_start = Slider(str_slider, 'Start time', 0,
                     total_time, valinit=500)
    pa1_slider = fig.add_subplot(gs_2[3])
    s_pa1 = Slider(pa1_slider, '$g_{Kleak}$', 0, 12,
                   valinit=Solver.g_kleak)
    pa2_slider = fig.add_subplot(gs_2[4])
    s_pa2 = Slider(pa2_slider, '$g_{Naleak}$', 0, 12,
                   valinit=Solver.g_naleak)
    pa3_slider = fig.add_subplot(gs_2[5])
    s_pa3 = Slider(pa3_slider, '$g_{Kleak}$ coupled', 0,
                   12, valinit=Solver.g_kleak)
    pa4_slider = fig.add_subplot(gs_2[6])
    s_pa4 = Slider(pa4_slider, '$g_{Naleak}$ coupled', 0,
                   12, valinit=Solver.g_naleak)
    pa5_slider = fig.add_subplot(gs_2[7])
    s_pa5 = Slider(pa5_slider, '$g_{coupled}$', 0, 12,
                   valinit=0)
    start_amp_slider = fig.add_subplot(gs_2[8])
    s_sas = Slider(start_amp_slider, 'start amp', -30, 30,
                   valinit=s.step_initial)

    gs_3 = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_master[2, :])
    coupled = GridSpecFromSubplotSpec(
        2, 1, height_ratios=[6, 1], subplot_spec=gs_3[0],
        hspace=0.0)
    coupled_axes = fig.add_subplot(coupled[0])
    injected_axes = fig.add_subplot(coupled[1])
    gs_4 = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_3[1])
    bifurcation = fig.add_subplot(gs_4[0])
    
    gs_5 = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_4[1])
    nullcline_inj = fig.add_subplot(gs_5[0])
    nullcline_coupled = fig.add_subplot(gs_5[1])

    def on_change(val):
        """
        Callback for changing the sliders, prints data about
        the solver's progress and gives the current values
        that are being solved.
        """
        s.step_amplitude = s_amp.val
        s.step_start = s_start.val
        s.step_duration = s_dur.val
        s.g_kleak = s_pa1.val
        s.g_naleak = s_pa2.val
        s.g_kleak2 = s_pa3.val
        s.g_naleak2 = s_pa4.val
        s.ggj = s_pa5.val
        s.step_initial = s_sas.val
        
        print('Updated value, solving')
        print('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' %
              (s.step_amplitude, s.step_start,
               s.step_duration, s.g_kleak,
               s.g_naleak, s.g_kleak2,
               s.g_naleak2, s.ggj))
        
        solve = odeint(s.coupled_system,
                   [s.Vrest, s.Vrest,
                    s.hTrest, s.hTrest], t)
        
        coupled_axes.clear()
        injected_axes.clear()
        bifurcation.clear()

        coupled_axes.plot(t, solve[:,0], 'k',
                          label='V clamped', linewidth=0.75)
        coupled_axes.plot(t, solve[:,1], 'k--',
                          label='V coupled', linewidth=0.75)
        coupled_axes.set_ylabel('voltage (mV)')
        coupled_axes.set_title('Coupled Voltage vs. Time')
        coupled_axes.legend(loc=1)
        
        injected_axes.plot(t, s.I_inj(t), 'k')
        injected_axes.set_xlabel('time (ms)')
        
        bifurcation.plot(s.g_kleak, s.g_naleak,
                         marker='^', label='injected',
                         color='k', markersize=10.0)
        bifurcation.plot(s.g_kleak2, s.g_naleak2,
                         marker='o', label='coupled',
                         color='k', markersize=10.0)
        bifurcation.plot(
            (s.g_kleak + s.g_kleak2)/2,
            (s.g_naleak + s.g_naleak2)/2,
            color='k', marker='x', label='average',
            markersize=15.0)
        bifurcation.plot(
            [s.g_kleak, s.g_kleak2],
            [s.g_naleak, s.g_naleak2],
            color='k', linewidth=0.85, linestyle='--')
        bifurcation.legend(loc=7)
        bifurcation.set_xlim([0, 12])
        bifurcation.set_ylim([0, 12])
        bifurcation.set_title(
            '$g_{K leak}$ vs. $g_{Na leak}$')
        bifurcation.set_xlabel('$g_{K leak}$')
        bifurcation.set_ylabel('$g_{Na leak}$')

        # Also solve for the nullclines
        nullcline_inj.clear()
        nullcline_coupled.clear()
        
        v = sp.linspace(-100, -30, 5000)
        # nullcline_inj.plot(
        #     v, s.injected_V_nullcline(v, s.Vrest,
        #                               s.step_initial),
        #     'k', linewidth=0.75, label='V')
        v1_min = min(
            solve[int(fidelity/3):2*int(fidelity/3), 0])
        v1_max = max(
            solve[int(fidelity/3):2*int(fidelity/3), 0])
        v2_min = min(
            solve[int(fidelity/3):2*int(fidelity/3), 1])
        v2_max = max(
            solve[int(fidelity/3):2*int(fidelity/3), 1])

        nullcline_inj.plot(
            v, s.injected_V_nullcline(
                v, v2_min, injection=s.step_initial),
            'k', linewidth=0.75, label='$\min(V_2)$')
        nullcline_inj.plot(
            v, s.injected_V_nullcline(
                v, v2_max, injection=s.step_initial),
            color='red', linewidth=0.75, label='$\max(V_2)$')
        # nullcline_inj.plot(
        #     v, s.injected_V_nullcline(v, s.Vrest,
        #                               s.step_initial + \
        #                               s.step_amplitude),
        #     'k--', linewidth=0.75, label='V injected')
        
        nullcline_inj.plot(
            v, s.h_Tinf(v), 'k:', linewidth=0.75,
            label='$h_{T\\infty}$')
        nullcline_inj.plot(
            solve[0:int(s.step_start), 0],
            solve[0:int(s.step_start), 2],
            color='blue', label='No injection')
        nullcline_inj.plot(
            solve[
                int(s.step_start):\
                int(s.step_start + s.step_duration), 0],
            solve[
                int(s.step_start):\
                int(s.step_start + s.step_duration), 2],
            color='green', label='Injected')
        nullcline_inj.plot(
            solve[int(s.step_start + s.step_duration):, 0],
            solve[int(s.step_start + s.step_duration):, 2],
            color='blue')
        nullcline_inj.legend(loc=1)
        
        # nullcline_coupled.plot(
        #     v, s.coupled_V_nullcline(s.Vrest, v),
        #     'k', linewidth=0.75)
        nullcline_coupled.plot(
            v, s.coupled_V_nullcline(
                v1_min, v, injection=s.step_initial),
            'k', linewidth=0.75, label='$\min(V_1)$')
        nullcline_coupled.plot(
            v, s.coupled_V_nullcline(
                v1_max, v, injection=s.step_initial),
            color='red', linewidth=0.75, label='$\max(V_1)$')
        nullcline_coupled.plot(solve[:,1], solve[:,3])
        nullcline_coupled.plot(
            v, s.h_Tinf(v), 'k:', linewidth=0.75,
            label='$h_{T\\infty}$')
        nullcline_coupled.legend(loc=1)

        nullcline_inj.set_ylim([-0.01, 0.2])
        nullcline_coupled.set_ylim([-0.01, 0.2])
        
        print("Solved")
        fig.canvas.draw()

    # Setting slider callbacks
    s_amp.on_changed(on_change)
    s_start.on_changed(on_change)
    s_dur.on_changed(on_change)
    s_pa1.on_changed(on_change)
    s_pa2.on_changed(on_change)
    s_pa3.on_changed(on_change)
    s_pa4.on_changed(on_change)
    s_pa5.on_changed(on_change)
    s_sas.on_changed(on_change)

    on_change(0)

def coupled_system_grid2(total_time=10000,
                        fidelity=10000):
    """ 
    Build a grid interface for looking at the dynamics of a
    coupled system and its associated bifurcation steady
    state plot. Also defines the update callback for updating
    the display with the new dynamics of the system.

    Caller must call plt.show() to display the graph.

    @param total_time: The time over which the interface will
    be integrating over. Time is measured in ms. Defaults to 
    10,000 (100 seconds).
    @param fidelity: The number of samples over the time.
    Defaults to 10,000, or 1 step per ms.
    @return: None
    """
    s = CoupledSolver()
    s.step_type = 'drop'
    s.step_initial = -10
    s.c = 'mh'
    
    if s.c is 'mh':
        # Change the resting voltage in the mh scheme
        # TODO: Add this to the table?
        s.Vrest = -64
        
    inj_amp = -4
    inj_start = 0
    inj_dur = 3000
    
    t = sp.linspace(0, total_time, fidelity)
    fig = plt.figure(figsize=(15,8))
    gs_master = GridSpec(3, 2, height_ratios=[0.1, 0.3, 10],
                         wspace=0.10, hspace=0.25, top=0.95,
                         bottom=0.075)
    gs_1 = GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs_master[0, :])
    title_axes = fig.add_subplot(gs_1[0])
    title_axes.set_title(
        'Coupled IT Ih Model Neurons', fontsize=30)
    title_axes.axis('off')

    gs_2 = GridSpecFromSubplotSpec(
        1, 9, subplot_spec=gs_master[1, :])

    amp_slider = fig.add_subplot(gs_2[0])
    s_amp = Slider(amp_slider, 'Injection Amplitude', -100,
                   10, valinit=inj_amp)
    dur_slider = fig.add_subplot(gs_2[1])
    s_dur = Slider(dur_slider, 'Duration', 0, total_time,
                   valinit=0)
    str_slider = fig.add_subplot(gs_2[2])
    s_start = Slider(str_slider, 'Start time', 0,
                     total_time, valinit=500)
    pa1_slider = fig.add_subplot(gs_2[3])
    s_pa1 = Slider(pa1_slider, '$g_{Kleak}$', 0, 12,
                   valinit=Solver.g_kleak)
    pa2_slider = fig.add_subplot(gs_2[4])
    s_pa2 = Slider(pa2_slider, '$g_{Naleak}$', 0, 12,
                   valinit=Solver.g_naleak)
    pa3_slider = fig.add_subplot(gs_2[5])
    s_pa3 = Slider(pa3_slider, '$g_{Kleak}$ coupled', 0,
                   12, valinit=Solver.g_kleak)
    pa4_slider = fig.add_subplot(gs_2[6])
    s_pa4 = Slider(pa4_slider, '$g_{Naleak}$ coupled', 0,
                   12, valinit=Solver.g_naleak)
    pa5_slider = fig.add_subplot(gs_2[7])
    s_pa5 = Slider(pa5_slider, '$g_{coupled}$', 0, 12,
                   valinit=0)
    start_amp_slider = fig.add_subplot(gs_2[8])
    s_sas = Slider(start_amp_slider, 'start amp', -30, 30,
                   valinit=s.step_initial)

    gs_3 = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_master[2, :])
    coupled = GridSpecFromSubplotSpec(
        2, 1, height_ratios=[6, 1], subplot_spec=gs_3[0],
        hspace=0.0)
    coupled_axes = fig.add_subplot(coupled[0])
    injected_axes = fig.add_subplot(coupled[1])
    gs_4 = GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs_3[1])
    nullcline_inj = nullcline_coupled = fig.add_subplot(gs_4[0])

    def on_change(val):
        """
        Callback for changing the sliders, prints data about
        the solver's progress and gives the current values
        that are being solved.
        """
        s.step_amplitude = s_amp.val
        s.step_start = s_start.val
        s.step_duration = s_dur.val
        s.g_kleak = s_pa1.val
        s.g_naleak = s_pa2.val
        s.g_kleak2 = s_pa3.val
        s.g_naleak2 = s_pa4.val
        s.ggj = s_pa5.val
        s.step_initial = s_sas.val
        
        print('Updated value, solving')
        print('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' %
              (s.step_amplitude, s.step_start,
               s.step_duration, s.g_kleak,
               s.g_naleak, s.g_kleak2,
               s.g_naleak2, s.ggj))
        
        solve = odeint(s.coupled_system,
                   [s.Vrest, s.Vrest,
                    s.hTrest, s.hTrest], t)
        
        coupled_axes.clear()
        injected_axes.clear()
        # bifurcation.clear()

        coupled_axes.plot(t, solve[:,0], 'k',
                          label='V clamped', linewidth=0.75)
        coupled_axes.plot(t, solve[:,1], 'k--',
                          label='V coupled', linewidth=0.75)
        coupled_axes.set_ylabel('voltage (mV)')
        coupled_axes.set_title('Coupled Voltage vs. Time')
        coupled_axes.legend(loc=1)
        
        injected_axes.plot(t, s.I_inj(t), 'k')
        injected_axes.set_xlabel('time (ms)')

        # Also solve for the nullclines
        nullcline_inj.clear()
        nullcline_coupled.clear()
        
        v = sp.linspace(-100, -20, 5000)
        v1_min = min(
            solve[int(fidelity/3):2*int(fidelity/3), 0])
        v1_max = max(
            solve[int(fidelity/3):2*int(fidelity/3), 0])
        v2_min = min(
            solve[int(fidelity/3):2*int(fidelity/3), 1])
        v2_max = max(
            solve[int(fidelity/3):2*int(fidelity/3), 1])

        nullcline_inj.plot(
            v, s.injected_V_nullcline(
                v, v2_min, injection=s.step_initial),
            color='', linestyle='--', linewidth=0.75, label='$\min(V_2)$')
        nullcline_inj.plot(
            v, s.injected_V_nullcline(
                v, v2_max, injection=s.step_initial),
            color='blue', linestyle=':', linewidth=0.75, label='$\max(V_2)$')

        nullcline_inj.plot(
            v, s.h_Tinf(v), 'k:', linewidth=0.75,
            label='$h_{T\\infty}$')
        nullcline_inj.plot(
            solve[0:int(s.step_start), 0],
            solve[0:int(s.step_start), 2],
            color='blue', label='No injection')
        nullcline_inj.plot(
            solve[
                int(s.step_start):\
                int(s.step_start + s.step_duration), 0],
            solve[
                int(s.step_start):\
                int(s.step_start + s.step_duration), 2],
            color='green', label='Injected')
        nullcline_inj.plot(
            solve[int(s.step_start + s.step_duration):, 0],
            solve[int(s.step_start + s.step_duration):, 2],
            color='blue')
        nullcline_inj.legend(loc=1)
        
        # nullcline_coupled.plot(
        #     v, s.coupled_V_nullcline(s.Vrest, v),
        #     'k', linewidth=0.75)
        nullcline_coupled.plot(
            v, s.coupled_V_nullcline(
                v1_min, v, injection=s.step_initial),
            color='green', linestyle='--', linewidth=0.75, label='$\min(V_1)$')
        nullcline_coupled.plot(
            v, s.coupled_V_nullcline(
                v1_max, v, injection=s.step_initial),
            color='green', linestyle=':', linewidth=0.75, label='$\max(V_1)$')
        nullcline_coupled.plot(solve[:,1], solve[:,3], color='green')
        nullcline_coupled.plot(
            v, s.h_Tinf(v), 'k:', linewidth=0.75,
            label='$h_{T\\infty}$')
        nullcline_coupled.legend(loc=1)

        nullcline_inj.set_ylim([-0.01, 0.2])
        nullcline_inj.set_xlim([-80, -20])
        nullcline_coupled.set_ylim([-0.01, 0.2])
        
        print("Solved")
        fig.canvas.draw()

    # Setting slider callbacks
    s_amp.on_changed(on_change)
    s_start.on_changed(on_change)
    s_dur.on_changed(on_change)
    s_pa1.on_changed(on_change)
    s_pa2.on_changed(on_change)
    s_pa3.on_changed(on_change)
    s_pa4.on_changed(on_change)
    s_pa5.on_changed(on_change)
    s_sas.on_changed(on_change)

    on_change(0)

if __name__ == "__main__":
    """ Run the grid when called by name """
    coupled_system_grid2()
    plt.show()
