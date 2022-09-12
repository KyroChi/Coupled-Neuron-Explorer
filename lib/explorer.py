#!/usr/local/bin/python3
""" PYTHON 3
This utility is designed to graphically display the dynamics
involved in coupled cells. 

Written by Kyle Chickering and Professor Timothy Lewis, 
Department of Mathematics at the University of California 
at Davis.

Copyright University of California at Davis.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as mplanim
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import (GridSpec, 
                                 GridSpecFromSubplotSpec)
from matplotlib.animation import FuncAnimation

import scipy as sp
from scipy.integrate import odeint
from lib.coupled import CoupledSolver

class Grid(object):
    """ 
    Build and manage the explorer grid.
    """
    objects = {
        'voltage_injected': None,
        'voltage_coupled': None,
        'amps_injected': None,
        'hTinf1': None,
        'hTinf2': None,
        'nullcline_injected': None,
        'nullcline_coupled': None,

        'voltage_injected_point': None,
        'voltage_coupled_point': None,
        'amps_injected_point': None,
        'nullcline_injected_point': None,
        'nullcline_coupled_point': None,

        'nullcline_injected_phase': None,
        'nullcline_coupled_phase': None,
        'nullcline_injected_phase_point': None,
        'nullcline_coupled_phase_point': None
    }
    sliders = []
    buttons = [
        [ 'back', None ],
        [ 'pause', None ],
        [ 'continue', None ],
        [ 'reset', None ],
        [ 'forward', None ]
    ]
    
    def __init__(self, fig, slider_list, *args,
                 **kwargs):
        """ """
        height_ratios = (kwargs.get('height_ratios') or
                         (0.1, 10))
        wspace = kwargs.get('wspace') or 0.10
        hspace = kwargs.get('hspace') or 0.25
        top = kwargs.get('top') or 0.95
        bottom = kwargs.get('bottom') or 0.075
        title = (kwargs.get('title') or 
                "Interactive Coupled Neurons")
        linewidth = kwargs.get('linewidth') or 0.8
        color = kwargs.get('color') or 'black'
        point_color = kwargs.get('point_color') or 'green'
        point_size = kwargs.get('point_size') or 7
        point_style = kwargs.get('point_style') or 'o'
        total_time = kwargs.get('total_time') or 10000
        voltage_range = (kwargs.get('voltage_range') or 
                        (-100, 20))
        ht_range = kwargs.get('ht_range') or (-0.01, 0.25)

        # Get the correct number of sliders and buttons
        control_length = len(slider_list) + 1
        
        gs_master = GridSpec(
            2, 1, height_ratios=height_ratios, wspace=wspace,
            hspace=hspace, top=top, bottom=bottom)
        # Row one is the title row
        gs_row_1 = GridSpecFromSubplotSpec(
            1, 1, subplot_spec=gs_master[0, :])
        # Row two is the row with plots and sliders
        gs_row_2 = GridSpecFromSubplotSpec(
            1, 2, width_ratios=(3, 1),
            subplot_spec=gs_master[1, :])

        title_axes = fig.add_subplot(gs_row_1[0])
        title_axes.set_title(title, fontsize=30)
        title_axes.axis('off')

        # Grid for all of the plotting
        gs_plotting = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_row_2[:, 0])
        # Nullcline plots
        gs_nullclines = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_plotting[1, :])
        # The voltage and the injection plot
        gs_voltage = GridSpecFromSubplotSpec(
            2, 1, hspace=0, height_ratios=(10, 2),
            subplot_spec=gs_plotting[0, :])

        voltage_axis = fig.add_subplot(gs_voltage[0, :])
        self.objects['voltage_injected'] = \
            voltage_axis.plot(
                [], [], linewidth=linewidth, color=color,
                linestyle='-', label='Voltage Clamped (mV)'
            )
        self.objects['voltage_injected_point'] = \
            voltage_axis.plot(
                [], [], color=point_color,
                markersize=point_size,
                marker=point_style
            )
        self.objects['voltage_coupled'] = \
            voltage_axis.plot(
                [], [], linewidth=linewidth, color=color,
                linestyle='--', label='Voltage Coupled (mV)'
            )
        self.objects['voltage_coupled_point'] = \
            voltage_axis.plot(
                [], [], color=point_color,
                markersize=point_size,
                marker=point_size
            )
        voltage_axis.set_xlim([0, total_time])
        voltage_axis.set_ylim(voltage_range)
        voltage_axis.legend(loc=1)
        
        injected_axis = fig.add_subplot(gs_voltage[1, :])
        self.objects['amps_injected'] = \
            injected_axis.plot(
                [], [], linewidth=linewidth, color=color,
                linestyle='--',
                label='Amperage Injected (pA)'
            )
        self.objects['amps_injected_point'] = \
            injected_axis.plot(
                [], [], color=point_color,
                markersize=point_size,
                marker=point_size
            )
        injected_axis.legend(loc=1)
        injected_axis.set_xlim([0, total_time])
        injected_axis.set_ylim([-50, 20])
        
        injected_nullcline_axis = \
                fig.add_subplot(gs_nullclines[0])
        self.objects['hTinf1'] = \
            injected_nullcline_axis.plot(
                [], [], linewidth=linewidth, color=color,
                linestyle='--', label='$h_{T\infty}$'
            )
        self.objects['nullcline_injected'] = \
            injected_nullcline_axis.plot(
                [], [], linewidth=linewidth, color=color,
                linestyle='-', label='$dV_1/dt = 0$'
            )
        self.objects['nullcline_injected_point'] = \
            injected_nullcline_axis.plot(
                [], [], color='red',
                markersize=point_size,
                marker='x'
            )
        self.objects['nullcline_injected_phase'] = \
            injected_nullcline_axis.plot(
                [], [], linewidth=linewidth, color='blue',
                linestyle='--'
            )
        self.objects['nullcline_injected_phase_point'] = \
            injected_nullcline_axis.plot(
                [], [], color='blue',
                markersize=point_size,
                marker=point_style
            )
        injected_nullcline_axis.legend(loc=1)
        injected_nullcline_axis.set_xlim(voltage_range)
        injected_nullcline_axis.set_ylim(ht_range)
        
        coupled_nullcline_axis = \
                fig.add_subplot(gs_nullclines[1])
        self.objects['hTinf2'] = \
            coupled_nullcline_axis.plot(
                [], [], linewidth=linewidth, color=color,
                linestyle='--', label='$h_{T\infty}$'
            )
        self.objects['nullcline_coupled'] = \
            coupled_nullcline_axis.plot(
                [], [], linewidth=linewidth, color=color,
                linestyle='-', label='$dV_2/dt = 0$'
            )
        self.objects['nullcline_coupled_point'] = \
            coupled_nullcline_axis.plot(
                [], [], color='red',
                markersize=point_size,
                marker='x'
            )
        self.objects['nullcline_coupled_phase'] = \
            coupled_nullcline_axis.plot(
                [], [], linewidth=linewidth, color='blue',
                linestyle='--'
            )
        self.objects['nullcline_coupled_phase_point'] = \
            coupled_nullcline_axis.plot(
                [], [], color='blue',
                markersize=point_size,
                marker=point_style
            )
        coupled_nullcline_axis.legend(loc=1)
        coupled_nullcline_axis.set_xlim(voltage_range)
        coupled_nullcline_axis.set_ylim(ht_range)

        # Grid for the sliders and animation controls
        gs_control = GridSpecFromSubplotSpec(
            control_length, 1, hspace=0.25,
            subplot_spec=gs_row_2[:, 1])

        # Build the sliders
        for i in range(len(slider_list)):
            self.sliders.append(
                [
                    slider_list[i],
                    fig.add_subplot(gs_control[i])
                ]
            )

        # Build the buttons
        gs_buttons = GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs_control[i + 1]
        )
        
        for i in range(len(self.buttons)):
            tmp = fig.add_subplot(gs_buttons[i])
            self.buttons[i][1] = Button(
                tmp, self.buttons[i][0]
            )
                
            self.buttons[i][1].hovercolor = 'green'
                                        
        return

class Explorer(object):
    """ """
    grid = None

    animation_on = True
    time = 0
    dt = 5

    total_time = 10000
    fidelity = 10000
    time_range = sp.linspace(0, total_time, fidelity)
    voltage_range = sp.linspace(-100, 20, 1000)
    solver = None

    # Store the data to improve animation speed
    data = {
        'voltage_injected': None,
        'voltage_coupled': None,
        'amps_injected': None,
        'hT1': None,
        'hT2': None,
    }

    fig = plt.figure(figsize=(15,10))
    
    def __init__(self, solver=None):
        slider_list = [
                ['Initial Voltage', (-100, 30, 3)],
                ['Step Amplitude', (-500, 10, 0)],
                ['Step Start', (0, 5000, 1442)],
                ['Step Duration', (0, 5000, 2615)],
                ['Coupling Constant', (0, 10, 9.51)],
                ['H-conductence', (0, 30, 2.18)],
            [r'$m_h \:\zeta$',(1, 500, 1.32)],
            [r'$m_h \:\xi$',(0.1, 200, 5.49)],
            [r'$m_T \:\chi$', (1, 500, 3)],
            ['K leak', (0, 10, 9.62)],
            ['Na leak', (0, 10, 3.04)],
            ['p_t', (0, 1.8*10**5, 7*10**4)],
            [r'$\beta$', (0, 1, 0.5)],
            [r'$1/\tau_h$', (0, 50, 1)],
            ]
        self.grid = Grid(self.fig, slider_list, total_time=self.total_time)

        self.solver = solver or CoupledSolver()
        self.solver.step_type = 'drop'
        
        # Set up the sliders and the button callbacks
        objects = self.grid.objects
        sliders = self.grid.sliders
        buttons = self.grid.buttons
        
        self.build_buttons(buttons)
        self.build_sliders(slider_list, sliders)

        self.plot_static()

        anim = FuncAnimation(
            self.fig, self.animate, interval=0.25,
            blit=True, init_func=self.initialize_animation,
            repeat=False
        )

        plt.show()
        return

    def build_buttons(self, buttons):
        """
        Build and link the buttons in the grid
        """
        for button in buttons:
            if button[0] == 'back':
                button[1].on_clicked(
                    self.step_animation_backward)
            elif button[0] == 'start':
                button[1].on_clicked(self.start_animation)
            elif button[0] == 'pause':
                button[1].on_clicked(self.pause_animation)
            elif button[0] == 'reset':
                button[1].on_clicked(self.reset_animation)
            else:
                button[1].on_clicked(
                    self.step_animation_forward)
                
        return

    def build_sliders(self, sliders_list, sliders):
        """ 
        Build the sliders that are returned by the grid 
        """
        for i in range(len(sliders_list)):
            print("Setting up slider")
            sliders[i][1] = Slider(
                sliders[i][1], sliders_list[i][0],
                sliders_list[i][1][0], sliders_list[i][1][1],
                valinit=sliders_list[i][1][2]
            )
            sliders[i][1].on_changed(self.slider_changed)

        return

    def get_time(self):
        """ 
        Get the current time step for the animation 
        """
        if self.animation_on:
            return (self.time + self.dt) % self.fidelity
        return self.time

    def slider_changed(self, val):
        """ """
        s = self.solver
        s.step_initial, s.step_amplitude, s.step_start, \
            s.step_duration, s.ggj, s.gbar_h, s.zeta, s.xi, s.chi, s.g_kleak, s.g_naleak, s.p_T, s.beta, s.one_tauh = (
                slider[1].val for slider in self.grid.sliders
            )

        self.plot_static()
        self.time = 0
        
        print("val is {}".format(val))
        return

    def plot_static(self):
        """ """
        s = self.solver
        
        solve = odeint(
            s.coupled_system_2,
            (s.Vrest, s.Vrest, s.hTrest, s.hTrest, s.mhrest, s.mhrest),
            self.time_range
        )

        self.data['voltage_injected'] = solve[:,0]
        self.data['voltage_coupled'] = solve[:,1]
        self.data['hT1'] = solve[:,2]
        self.data['hT2'] = solve[:,3]

        self.data['amps_injected'] = \
            s.I_inj(self.time_range)

        self.grid.objects['voltage_injected'][0].set_data(
            self.time_range, self.data['voltage_injected']
        )
        self.grid.objects['voltage_coupled'][0].set_data(
            self.time_range, self.data['voltage_coupled']
        )
        self.grid.objects['amps_injected'][0].set_data(
            self.time_range, self.data['amps_injected']
        )
        self.grid.objects['hTinf1'][0].set_data(
            self.voltage_range, s.h_Tinf(self.voltage_range)
        )
        self.grid.objects['hTinf2'][0].set_data(
            self.voltage_range, s.h_Tinf(self.voltage_range)
        )
        return

    def initialize_animation(self):
        """ """
        return self.return_grid_objects()

    def animate(self, i):
        """ 
        """
        self.time = self.get_time()
        self.grid.objects[
            'amps_injected_point'][0].set_data(
                self.time, self.solver.I_inj(self.time)
            )
        self.grid.objects[
            'voltage_injected_point'][0].set_data(
                self.time,
                self.data['voltage_injected'][self.time]
        )
        self.grid.objects[
            'voltage_coupled_point'][0].set_data(
                self.time,
                self.data['voltage_coupled'][self.time]
        )
        self.grid.objects['amps_injected_point'][0].set_data(
            self.time, self.data['amps_injected'][self.time]
        )

        self.grid.objects['nullcline_injected'][0].set_data(
            self.voltage_range,
            self.solver.injected_V_nullcline(
                self.voltage_range,
                self.data['voltage_coupled'][self.time]
            )
        )
        self.grid.objects['nullcline_coupled'][0].set_data(
            self.voltage_range,
            self.solver.coupled_V_nullcline(
                self.data['voltage_injected'][self.time],
                self.voltage_range
            )
        )

        self.grid.objects[
            'nullcline_injected_phase'][0].set_data(
                self.data['voltage_injected'][0:self.time],
                self.data['hT1'][0:self.time]
        )
        self.grid.objects[
            'nullcline_coupled_phase'][0].set_data(
                self.data['voltage_coupled'][0:self.time],
                self.data['hT2'][0:self.time]
        )

        self.grid.objects[
            'nullcline_injected_phase_point'][0].set_data(
                self.data['voltage_injected'][self.time],
                self.data['hT1'][self.time]
            )
        self.grid.objects[
            'nullcline_coupled_phase_point'][0].set_data(
                self.data['voltage_coupled'][self.time],
                self.data['hT2'][self.time]
            )

        inj_int = sp.argwhere(
            sp.diff(sp.sign(self.solver.injected_V_nullcline(
                self.voltage_range,
                self.data['voltage_coupled'][self.time]
            ) - self.solver.h_Tinf(self.voltage_range)))
        )
        self.grid.objects[
            'nullcline_injected_point'][0].set_data(
                self.voltage_range[inj_int],
                self.solver.h_Tinf(
                    self.voltage_range)[inj_int]
            )

        inj_int = sp.argwhere(
            sp.diff(sp.sign(self.solver.coupled_V_nullcline(
                self.data['voltage_injected'][self.time],
                self.voltage_range
            ) - self.solver.h_Tinf(self.voltage_range)))
        )
        self.grid.objects[
            'nullcline_coupled_point'][0].set_data(
                self.voltage_range[inj_int],
                self.solver.h_Tinf(
                    self.voltage_range)[inj_int]
            )
            
        return self.return_grid_objects()
    
    def return_grid_objects(self):
        """ 
        Returns all of the grid objects which we need for the
        animator.
        """
        return self.grid.objects['voltage_injected'][0], \
            self.grid.objects['voltage_coupled'][0], \
            self.grid.objects['amps_injected'][0], \
            self.grid.objects['hTinf1'][0], \
            self.grid.objects['hTinf2'][0], \
            self.grid.objects['nullcline_injected'][0], \
            self.grid.objects['nullcline_coupled'][0], \
            self.grid.objects['voltage_injected_point'][0], \
            self.grid.objects['voltage_coupled_point'][0], \
            self.grid.objects['amps_injected_point'][0], \
            self.grid.objects[
                'nullcline_injected_point'][0], \
            self.grid.objects['nullcline_coupled_point'][0], \
            self.grid.objects[
                'nullcline_injected_phase'][0], \
            self.grid.objects['nullcline_coupled_phase'][0], \
            self.grid.objects[
                'nullcline_injected_phase_point'][0], \
            self.grid.objects[
                'nullcline_coupled_phase_point'][0]
        
    def start_animation(self, event):
        self.animation_on = True
        return

    def pause_animation(self, event):
        self.animation_on = False
        return

    def reset_animation(self, event):
        self.time = 0
        return

    def step_animation_forward(self, event):
        self.time += self.dt
        return

    def step_animation_backward(self, event):
        self.time += self.dt
        return

if __name__ == "__main__":
    e = Explorer()
