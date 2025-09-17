from brian2 import *
from equations import *
from connectivity import build_pen_to_epg_indices, build_pen_to_epg_array
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
# set_device('cpp_standalone', build_on_run=False)


def visual_cue(theta, index, stimulus = 0.03, sigma = 2 * np.pi/8):
    """
    param: 
    theta: the angle of the visual input
    index: the index of the neuron
    stimulus: the strength of the visual input
    sigma: the standard deviation of the Gaussian distribution
    """
    A = stimulus
    phi = index * np.pi/8
    d1 = (theta-phi)**2 
    d2 = (theta-phi + 2*np.pi)**2
    d3 = (theta-phi - 2*np.pi)**2
    return A * (np.exp(-d1/(2*sigma**2)) + np.exp(-d2/(2*sigma**2)) + np.exp(-d3/(2*sigma**2)))

def simulator( 
        # parameters
        w_EE = 0.719, # EB <-> EB
        w_EI = 0.143, # R -> EB
        w_IE = 0.740, # EB -> R
        w_II = 0.01, # R <-> R
        w_PP = 0.01, # PEN <-> PEN
        w_EP = 0.012, # EB -> PEN 
        w_PE = 0.709, # PEN -> EB
        sigma = 0.0001, # noise level
        
        stimulus_strength = 0.05, 
        stimulus_location = 0*np.pi, # from 0 to np.pi
        shifter_strength = 0.015,
        half_PEN = 'right',
        
        t_epg_open = 200, # stimulus
        t_epg_close = 500,    # no stimulus
        t_pen_open = 1000,   # shift

):
    """Simulate the head direction network with visual cues and body rotation."""

    start = time.time()
    start_scope()  
    
    taum   = 20*ms   # time constant
    Cm     = 0.1
    g_L    = 10   # leak conductance
    E_l    = -0.07  # leak reversal potential (volt)
    E_e    = 0   # excitatory reversal potential
    tau_e  = 5*ms    # excitatory synaptic time constant
    Vr     = E_l     # reset potential
    Vth    = -0.05  # spike threshold (volt)
    Vs     = 0.02   # spiking potential (volt)
    w_e    = 0.1  	 # excitatory synaptic weight (units of g_L)
    v_e    = 5*Hz    # excitatory Poisson rate
    N_e         = 100     # number of excitatory inputs
    E_ach       = 0
    tau_ach     = 10*ms
    E_GABAA     = -0.07 # GABAA reversal potential
    tau_GABAA   = 5*ms # GABAA synaptic time constant

    # create neuron
    EPG = NeuronGroup(48, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler' )
    PEN = NeuronGroup(48,model=eqs_PEN, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')

    # initialize neuron1
    EPG.v = E_l
    PEN.v = E_l
    R.v = E_l

    EPG_groups = [EPG[i:i+3] for i in range(0, 48, 3)]
    PEN_groups = [PEN[i:i+3] for i in range(0, 48, 3)]
    R_groups = [R[0:3]]
    
    # ========= EPG -> EPG =========
    S_EE = Synapses(EPG, EPG, Ach_eqs, on_pre='s_ach += w_EE', method='euler')
    S_EE.connect(condition='i//3 == j//3 and i != j')

    # ========= PEN -> PEN =========
    S_PP = Synapses(PEN, PEN, Ach_eqs_PP, on_pre='s_ach += w_PP', method='euler')
    S_PP.connect(condition='i//3 == j//3 and i != j')
    
    # ========= EPG -> R =========
    S_EI = Synapses(EPG, R, model=Ach_eqs_EI, on_pre='s_ach += w_EI', method='euler')
    S_EI.connect(condition='True')
    
    # ========= EPG <- R =========
    S_IE = Synapses(R, EPG, model=GABA_eqs, on_pre='s_GABAA += w_IE', method='euler')
    S_IE.connect(condition='True')
    
    # ========= R -> R =========
    S_II = Synapses(R, R, model=GABA_eqs_i, on_pre='s_GABAA += w_II', method='euler')
    S_II.connect(condition='i != j')
    
    # ========= EPG -> PEN =========
    S_EP = Synapses(EPG, PEN, Ach_eqs_EP, on_pre='s_ach += w_EP', method='euler')
    S_EP.connect(condition='i//3 == j//3')
    
    # ========= PEN -> EPG (optimized by connectivity matrix) =========
    S_PE_2 = Synapses(PEN, EPG, model=Ach_eqs_PE_2, on_pre='s_ach += 2*w_PE', method='euler')
    S_PE_1 = Synapses(PEN, EPG, model=Ach_eqs_PE_1, on_pre='s_ach += 1*w_PE', method='euler')

    pre2, post2, pre1, post1 = build_pen_to_epg_array()

    # ---- build all connections at once ----
    S_PE_2.connect(i=pre2, j=post2)
    S_PE_1.connect(i=pre1, j=post1)
    print(f'{time.strftime("%H:%M:%S")} [info] All connections done')
    # ========= end PEN -> EPG =========
    # record model state

    PRMs = PopulationRateMonitor(EPG)
    PRM0 = PopulationRateMonitor(EPG_groups[0])
    PRM1 = PopulationRateMonitor(EPG_groups[1])
    PRM2 = PopulationRateMonitor(EPG_groups[2])
    PRM3 = PopulationRateMonitor(EPG_groups[3])
    PRM4 = PopulationRateMonitor(EPG_groups[4])
    PRM5 = PopulationRateMonitor(EPG_groups[5])
    PRM6 = PopulationRateMonitor(EPG_groups[6])
    PRM7 = PopulationRateMonitor(EPG_groups[7])
    PRM8 = PopulationRateMonitor(EPG_groups[8])
    PRM9 = PopulationRateMonitor(EPG_groups[9])
    PRM10 = PopulationRateMonitor(EPG_groups[10])
    PRM11 = PopulationRateMonitor(EPG_groups[11])
    PRM12 = PopulationRateMonitor(EPG_groups[12])
    PRM13 = PopulationRateMonitor(EPG_groups[13])
    PRM14 = PopulationRateMonitor(EPG_groups[14])
    PRM15 = PopulationRateMonitor(EPG_groups[15])

    PRM_PEN0 = PopulationRateMonitor(PEN_groups[0])
    PRM_PEN1 = PopulationRateMonitor(PEN_groups[1])
    PRM_PEN2 = PopulationRateMonitor(PEN_groups[2])
    PRM_PEN3 = PopulationRateMonitor(PEN_groups[3])
    PRM_PEN4 = PopulationRateMonitor(PEN_groups[4])
    PRM_PEN5 = PopulationRateMonitor(PEN_groups[5])
    PRM_PEN6 = PopulationRateMonitor(PEN_groups[6])
    PRM_PEN7 = PopulationRateMonitor(PEN_groups[7])
    PRM_PEN8 = PopulationRateMonitor(PEN_groups[8])
    PRM_PEN9 = PopulationRateMonitor(PEN_groups[9])
    PRM_PEN10 = PopulationRateMonitor(PEN_groups[10])
    PRM_PEN11 = PopulationRateMonitor(PEN_groups[11])
    PRM_PEN12 = PopulationRateMonitor(PEN_groups[12])
    PRM_PEN13 = PopulationRateMonitor(PEN_groups[13])
    PRM_PEN14 = PopulationRateMonitor(PEN_groups[14])
    PRM_PEN15 = PopulationRateMonitor(PEN_groups[15])
    
    PRM_R0 = PopulationRateMonitor(R_groups[0])
    
    net=Network(collect())
    net.add(S_EP,S_EE,S_PP,S_EI,S_IE,S_II)

    # run simulation

    ## SIMULATION ###

    stimulus_location %= 2*np.pi
    theta_r = stimulus_location/2
    theta_l = theta_r + np.pi
    A = stimulus_strength

    # for k in tqdm(range(0,100)):    
    #     for i in range(0,8):
    #         EPG_groups[i].I = visual_cue(theta_r, i, A)
    #     for i in range(8,16):
    #         EPG_groups[i].I = visual_cue(theta_l, i, A)
    #     net.run(10*ms)
    #     for i in range(0,16):
    #         EPG_groups[i].I = 0


    print(f'{time.strftime("%H:%M:%S")} [info] Testing stability')
    net.run(t_epg_open*ms)

    for i in range(0,16):
        EPG_groups[i].I = 0
    net.run(t_epg_close * ms)

    if half_PEN == 'right':
        for i in range(8):
            PEN_groups[i].I = shifter_strength
    elif half_PEN == 'left':
        for i in range(8,16):
            PEN_groups[i].I = shifter_strength

    print(f'{time.strftime("%H:%M:%S")} [info] Testing shifting simulation')
    net.run(t_pen_open * ms)
    
    end  = time.time()
    print(f'\r{time.strftime("%H:%M:%S")} : {(end - start)//60:.0f} min {(end - start)%60:.1f} sec -> eval end', flush=True)
    fr_epg = [PRM0.smooth_rate(width=5*ms),
                    PRM1.smooth_rate(width=5*ms),
                    PRM2.smooth_rate(width=5*ms),
                    PRM3.smooth_rate(width=5*ms),
                    PRM4.smooth_rate(width=5*ms),
                    PRM5.smooth_rate(width=5*ms),
                    PRM6.smooth_rate(width=5*ms),
                    PRM7.smooth_rate(width=5*ms),
                    PRM8.smooth_rate(width=5*ms),
                    PRM9.smooth_rate(width=5*ms),
                    PRM10.smooth_rate(width=5*ms),
                    PRM11.smooth_rate(width=5*ms),
                    PRM12.smooth_rate(width=5*ms),
                    PRM13.smooth_rate(width=5*ms),
                    PRM14.smooth_rate(width=5*ms),
                    PRM15.smooth_rate(width=5*ms),]
    
    fr_pen = [PRM_PEN0.smooth_rate(width=5*ms),
                    PRM_PEN1.smooth_rate(width=5*ms),
                    PRM_PEN2.smooth_rate(width=5*ms),
                    PRM_PEN3.smooth_rate(width=5*ms),
                    PRM_PEN4.smooth_rate(width=5*ms),
                    PRM_PEN5.smooth_rate(width=5*ms),
                    PRM_PEN6.smooth_rate(width=5*ms),
                    PRM_PEN7.smooth_rate(width=5*ms),
                    PRM_PEN8.smooth_rate(width=5*ms),
                    PRM_PEN9.smooth_rate(width=5*ms),
                    PRM_PEN10.smooth_rate(width=5*ms),
                    PRM_PEN11.smooth_rate(width=5*ms),
                    PRM_PEN12.smooth_rate(width=5*ms),
                    PRM_PEN13.smooth_rate(width=5*ms),
                    PRM_PEN14.smooth_rate(width=5*ms),
                    PRM_PEN15.smooth_rate(width=5*ms),]
    
    fr_r = [PRM_R0.smooth_rate(width=5*ms),]

    fr_epg = np.array(fr_epg)
    fr_pen = np.array(fr_pen)
    fr_r = np.array(fr_r)
    t = np.linspace(0, len(fr_epg[0])/10000, len(fr_epg[0]))


    return t, fr_epg, fr_pen, fr_r

if __name__ == '__main__':
    t, fr, fr_pen, fr_r = simulator()    
