import scipy as sp
 
class Solver(object):
    """
    Solves the associated systme of equations that governs
    the behavior of the Amarillo IT Ih leak model neurons.
    This object doesn't do anything special on startup.'
    """
    step_type = 'discontinuous'
    step_initial = 6
    step_start = 400
    step_duration = 600
    step_amplitude = -10 # 30 was getting there
    total_time = 3000
    fidelity = 10000
    Vrest = -61.5
    hTrest = 0.033
    # hTrest = 0.030

    c = '2d'
    C = 0.2
    S = 20000
    T = 36 + 273
    C_ao = 2 * 10**-3
    C_ai = 0.05 * 10**-6
    ggj = 1*10**-5
    g_kleak = 1*10**-5
    E_kleak = -100
    g_naleak = 3*10**-6
    E_naleak = 0
    gbar_h = 2.2 * 10**-5
    E_h = -43
    z = 2
    F = 96485.33289 # Faraday constant
    R = 8.314472 # Gas constant
    
    table2 = {
        '3d': \
        (7 * 10**-5, -53, -128, -12.8, -75, -461, -16),
        '2d': \
        (7 * 10**-5, -53, None, None, -75, -461, -16),
        'as': \
        (3 * 10**-5, -56, -131, -15.8, -75, -461, -16),
        'mh': \
        (1.1 * 10**-4, -57, -132, -16.8, -81, -467, -22),
    }
    
    def step(self, t):
        if self.step_type is "continuous":
            """ Return a double sigmoid step function """
            return self.step_amplitude /\
                ( sp.exp(-1000*(t - self.step_start))+ 1 ) - \
                self.step_amplitude / \
                ( sp.exp(-1000*(t - self.step_duration - \
                                self.step_start))+1 )
        
        elif self.step_type is "discontinuous":
            """ Discontinuous Heaviside step function """
            return self.step_amplitude * \
                sp.heaviside(t - self.step_start, 1) \
                - self.step_amplitude * \
                sp.heaviside(t - (self.step_start + self.step_duration)
                             , 1)
        
        elif self.step_type is "drop":
            """ Starts with a depolarizing current and drops
            the appropriate ampitude """
            return self.step_initial + \
                self.step_amplitude * \
                sp.heaviside(t - self.step_start, 1) \
                - self.step_amplitude * \
                sp.heaviside(t - (self.step_start + self.step_duration)
                             , 1)

    def m_Tinf (self, V):
        return 1 / ( 1 + sp.exp(
            (V - self.table2[self.c][1])/(-6.2) ) )

    def tau_mT (self, V):
        # TODO: This function looks super gross
        return (
            0.612 + 1/(
                sp.exp(-(
                    V - self.table2[self.c][2])/16.7) + \
                sp.exp((
                    V - self.table2[self.c][3])/18.2)))/3

    def h_Tinf (self, V):
        return 1 / ( 1 + sp.exp(
            (V - self.table2[self.c][4]) / 4) )

    def tau_hT (self, V):
        if V < -75:
            return sp.exp(
                (V - self.table2[self.c][5])/66.6)/3
        else:
            return (28 + sp.exp(
                (V - self.table2[self.c][6])/-10.5))/3
    
    def P_T (self, V):
        return self.table2[self.c][0]

    def m_hinf(self, V):
        return 1 / ( 1 + sp.exp( (V + 82 ) / 5.49 ) )

    def tau_mh(self, V):
        return (1/(0.0008 + 0.0000035*sp.exp(-0.05787*V) + \
                   sp.exp(-1.87 + 0.0701*V)))/1.32

    def G(self, V):
        """ Linearized calicium cation function """
        x1 = (self.z*self.F)/(self.R*self.T)
        xe2 = 1 - sp.exp(x1*V)
                
        # return x1*self.z*self.F * V * ( self.C_ao ) / ( xe2 )
        return x1*self.z*self.F*V*self.C_ao/(1-sp.exp(x1*V))

    def I_T(self, V,m_T, h_T):
        return self.table2[self.c][0] * m_T**2 * h_T * \
            self.S * self.G(V)

    def I_T_no_time (self, V, h_T):
        return self.table2[self.c][0] * \
            self.m_Tinf(V)**2 * h_T * self.S * self.G(V)

    def I_Kleak (self, V):
        return self.g_kleak * (V - self.E_kleak) * self.S

    def I_Naleak (self, V):
        return self.g_naleak * (V - self.E_naleak) * self.S

    def I_h (self, V, m_h):
        return self.gbar_h * m_h * (V - self.E_h) * self.S

    def I_inj (self, t):
        step_v = sp.vectorize( lambda t: self.step(t) )
        return step_v(t) # returns a numpy array

    def dmT_dt (self, V, m_T):
        return (self.m_Tinf(V) - m_T)/self.tau_mT(V)

    def dhT_dt (self, V, h_T):
        return (self.h_Tinf(V) - h_T)/(self.tau_hT(V))
    
    def dmh_dt (self, V, m_h):
        return (self.m_hinf(V) - m_h)/self.tau_mh(V)
