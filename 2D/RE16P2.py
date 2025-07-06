from brian2 import *
import numpy as np
import time
from equations_1Rv import *

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
        Isynmax = 0.2,
        Isynmax_1 = 0.2,
        IEImax = 1,
        w_EE = 0.719, # EB <-> EB
        w_EI = 0.143, # EPG -> R
        w_IE = 0.740, # R -> EPG
        w_II = 0.01, # R <-> R
        w_PP = 0.01, # PEN <-> PEN
        w_EP = 0.012, # EB -> PEN 
        w_PE = 0.709, # PEN -> EB
        sigma = 0.0001, # noise level
        
        stimulus_strength = 0.05, 
        stimulus_strength_1 = 0.05,
        stimulus_location = 0*np.pi, # from 0 to np.pi
        stimulus_location_1 = 0*np.pi,
        shifter_strength = 0.015,
        half_PEN = 'right',
        
        t_epg_open = 200, # stimulus
        t_epg_close = 500,    # no stimulus
        t_pen_open = 5000,   # shift

):
    """
    param:
    w_EE: the weight of the EPG to EPG synapse default 0.719
    w_EI: the weight of the EPG to R synapse default 0.143
    w_IE: the weight of the R to EPG synapse default 0.740
    w_II: the weight of the R to R synapse default 0.01
    w_PP: the weight of the PEN to PEN synapse default 0.01
    w_EP: the weight of the EPG to PEN synapse default 0.008
    w_PE: the weight of the PEN to EPG synapse default 0.811
    sigma: the noise level default 0.001
    stimulus_strength: the strength of the visual input default 0.05
    stimulus_location: the location of the visual input (from 0 to np.pi) default 0*np.pi
    stimulus_strength_1: the strength of the visual input default 0.05
    stimulus_location_1: the location of the visual input (from 0 to np.pi) default 0*np.pi
    shifter_strength: the strength of the shifter neuron input default 0.015
    ang_vel: the angular velocity of the cue rotation
    activate_duration: give the visual cue
    bump_duration: close the visual cue input
    shift_duration: the duration of the body ratotion
    half_PEN: 'left' or 'right'
    half_PEN_1: 'left' or 'right'
    t_epg_open: the duration of the visual cue input
    t_epg_close: the duration of the no visual cue input
    t_pen_open: the duration of the body ratotion
    R_refractory: the refractory period of the R neuron (unit: ms)
    
    """

    start = time.time()
    start_scope()  
    print('start')
    
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
    E_ach_1      = 0        # vertical ACh reversal potential
    tau_ach_1    = 10*ms    # vertical ACh synaptic time constant
    E_GABAA     = -0.07 # GABAA reversal potential
    tau_GABAA   = 5*ms # GABAA synaptic time constant
    E_GABAA_1    = -0.07    # vertical GABAA reversal potential  
    tau_GABAA_1  = 5*ms     # vertical GABAA synaptic time constant
    


    # create neuron
    EPG = NeuronGroup(48, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler' )
    PEN = NeuronGroup(48,model=eqs_PEN, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    EPG_1 = NeuronGroup(48, model=eqs_EPG_1, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler' )
    # PEN_1 = NeuronGroup(48,model=eqs_PEN_1, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler') # delete
    R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')

    # initialize neuron1
    EPG.v = E_l
    PEN.v = E_l
    EPG_1.v = E_l
    R.v = E_l

    EPG_groups = []
    EPG_groups.append(EPG[0:3]) # EPG1
    EPG_groups.append(EPG[3:6]) # EPG2
    EPG_groups.append(EPG[6:9]) # EPG3
    EPG_groups.append(EPG[9:12]) # EPG4
    EPG_groups.append(EPG[12:15]) # EPG5
    EPG_groups.append(EPG[15:18]) # EPG6
    EPG_groups.append(EPG[18:21]) # EPG7
    EPG_groups.append(EPG[21:24]) # EPG8
    EPG_groups.append(EPG[24:27]) # EPG9
    EPG_groups.append(EPG[27:30]) # EPG10
    EPG_groups.append(EPG[30:33]) # EPG11
    EPG_groups.append(EPG[33:36]) # EPG12
    EPG_groups.append(EPG[36:39]) # EPG13
    EPG_groups.append(EPG[39:42]) # EPG14
    EPG_groups.append(EPG[42:45]) # EPG15
    EPG_groups.append(EPG[45:48]) # EPG16

    EPG_1_groups = []
    EPG_1_groups.append(EPG_1[0:3]) # EPG1
    EPG_1_groups.append(EPG_1[3:6]) # EPG2
    EPG_1_groups.append(EPG_1[6:9]) # EPG3
    EPG_1_groups.append(EPG_1[9:12]) # EPG4
    EPG_1_groups.append(EPG_1[12:15]) # EPG5
    EPG_1_groups.append(EPG_1[15:18]) # EPG6 
    EPG_1_groups.append(EPG_1[18:21]) # EPG7
    EPG_1_groups.append(EPG_1[21:24]) # EPG8
    EPG_1_groups.append(EPG_1[24:27]) # EPG9
    EPG_1_groups.append(EPG_1[27:30]) # EPG10
    EPG_1_groups.append(EPG_1[30:33]) # EPG11
    EPG_1_groups.append(EPG_1[33:36]) # EPG12
    EPG_1_groups.append(EPG_1[36:39]) # EPG13
    EPG_1_groups.append(EPG_1[39:42]) # EPG14
    EPG_1_groups.append(EPG_1[42:45]) # EPG15
    EPG_1_groups.append(EPG_1[45:48]) # EPG16

    PEN_groups = []
    PEN_groups.append(PEN[0:3]) # PEN1
    PEN_groups.append(PEN[3:6]) # PEN2
    PEN_groups.append(PEN[6:9]) # PEN3
    PEN_groups.append(PEN[9:12]) # PEN4
    PEN_groups.append(PEN[12:15]) # PEN5
    PEN_groups.append(PEN[15:18]) # PEN6
    PEN_groups.append(PEN[18:21]) # PEN7
    PEN_groups.append(PEN[21:24]) # PEN8
    PEN_groups.append(PEN[24:27]) # PEN9
    PEN_groups.append(PEN[27:30]) # PEN10
    PEN_groups.append(PEN[30:33]) # PEN11
    PEN_groups.append(PEN[33:36]) # PEN12
    PEN_groups.append(PEN[36:39]) # PEN13
    PEN_groups.append(PEN[39:42]) # PEN14
    PEN_groups.append(PEN[42:45]) # PEN15
    PEN_groups.append(PEN[45:48]) # PEN16

    # PEN_1_groups = []

    # PEN_1_groups.append(PEN_1[0:3]) # PEN1
    # PEN_1_groups.append(PEN_1[3:6]) # PEN2
    # PEN_1_groups.append(PEN_1[6:9]) # PEN3
    # PEN_1_groups.append(PEN_1[9:12]) # PEN4
    # PEN_1_groups.append(PEN_1[12:15]) # PEN5
    # PEN_1_groups.append(PEN_1[15:18]) # PEN6
    # PEN_1_groups.append(PEN_1[18:21]) # PEN7
    # PEN_1_groups.append(PEN_1[21:24]) # PEN8
    # PEN_1_groups.append(PEN_1[24:27]) # PEN9
    # PEN_1_groups.append(PEN_1[27:30]) # PEN10
    # PEN_1_groups.append(PEN_1[30:33]) # PEN11
    # PEN_1_groups.append(PEN_1[33:36]) # PEN12
    # PEN_1_groups.append(PEN_1[36:39]) # PEN13
    # PEN_1_groups.append(PEN_1[39:42]) # PEN14
    # PEN_1_groups.append(PEN_1[42:45]) # PEN15
    # PEN_1_groups.append(PEN_1[45:48]) # PEN16

    EPG_syn = []
    PEN_syn = []
    PE2R_syn = []
    PE2L_syn = []
    PE1R_syn = []
    PE1L_syn = []
    PE2R_syn2 = []
    PE2L_syn2 = []
    PE1R_syn2 = []
    PE1L_syn2 = []

    EP_syn = []
    EP_1_syn = []

    EPG_1_syn = []
    PEN_1_syn = []
    PE2R_1_syn = []
    PE2L_1_syn = []
    PE1R_1_syn = []
    PE1L_1_syn = []
    PE2R_1_syn2 = []
    PE2L_1_syn2 = []
    PE1R_1_syn2 = []
    PE1L_1_syn2 = []
    PE7_1_syn = []
    PE8_1_syn = []

    
    # EPG_EPG
    print("Creating EPG-EPG connections...")
    for k in range(0,16):
        # EPG to EPG
        EPG_syn.append(Synapses(EPG_groups[k], EPG_groups[k], Ach_eqs, on_pre='s_ach += w_EE', method='euler'))
        EPG_syn[k].connect(condition='i != j')
    
    # PEN_PEN
    print("Creating PEN-PEN connections...")
    for k2 in range(0,16):
        # PEN to PEN
        PEN_syn.append(Synapses(PEN_groups[k2], PEN_groups[k2], Ach_eqs_PP, on_pre='s_ach += w_PP', method='euler'))
        PEN_syn[k2].connect(condition='i != j')
    
    # EPG_R and R_EPG
    # EPG to R
    print("Creating EPG-R connections...")
    S_EI = Synapses(EPG, R, model=Ach_eqs_EI, on_pre='s_ach += w_EI', method='euler')
    for a in range(0,48):
        for b in range(0,3):
            S_EI.connect(i=a, j=b)
    
    # R to EPG
    
    print("Creating R-EPG connections...")
    S_IE = Synapses(R, EPG, model=GABA_eqs, on_pre='s_GABAA += w_IE', method='euler')
    for a2 in range(0,48):
        for b2 in range(0,3):
            S_IE.connect(i=b2, j=a2)
    
    # R <-> R
    print("Creating R-R connections...")
    S_II = Synapses(R, R, model=GABA_eqs_i, on_pre='s_GABAA += w_II', method='euler')
    S_II.connect(condition='i != j')
    
    print("Creating EPG-PEN connections...")
    # EPG_PEN synapse
    for k3 in range(0,16):
        # EPG to PEN
        EP_syn.append(Synapses(EPG_groups[k3], PEN_groups[k3], Ach_eqs_EP, on_pre='s_ach += w_EP', method='euler'))
        EP_syn[k3].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    ###
    
    # PEN_EPG synapse #v
    # PEN0-6 -> EPG0-8
    for k4 in range(0,7):
        # PEN to EPG
        PE2R_syn.append(Synapses(PEN_groups[k4], EPG_groups[k4+1], Ach_eqs_PE2R, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2R_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1R_syn.append(Synapses(PEN_groups[k4], EPG_groups[k4+2], Ach_eqs_PE1R, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1R_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE1R_syn.append(Synapses(PEN_groups[6], EPG_groups[0], Ach_eqs_PE1R, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1R_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG0-8 
    for k4 in range(0,7):
        # PEN to EPG
        PE2R_syn2.append(Synapses(PEN_groups[k4+9], EPG_groups[k4+1], Ach_eqs_PE2R2, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2R_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1R_syn2.append(Synapses(PEN_groups[k4+9], EPG_groups[k4+2], Ach_eqs_PE1R2, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1R_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE1R_syn2.append(Synapses(PEN_groups[15], EPG_groups[0], Ach_eqs_PE1R2, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1R_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN7-8
    PE7_syn = []
    # PEN7 -> EPG connections
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[0], Ach_eqs_PE7, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[1], Ach_eqs_PE7, on_pre='s_ach += 1*w_PE', method='euler'))
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[15], Ach_eqs_PE7, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[14], Ach_eqs_PE7, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE7_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE8_syn = []
    # PEN8 -> EPG connections
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[0], Ach_eqs_PE8, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[1], Ach_eqs_PE8, on_pre='s_ach += 1*w_PE', method='euler'))
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[15], Ach_eqs_PE8, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[14], Ach_eqs_PE8, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE8_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN0-6 -> EPG8-15
    for k4 in range(0,7):
        # PEN to EPG
        PE2L_syn.append(Synapses(PEN_groups[k4], EPG_groups[k4+8], Ach_eqs_PE2L, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2L_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1L_syn.append(Synapses(PEN_groups[k4+1], EPG_groups[k4+8], Ach_eqs_PE1L, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1L_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
        
    PE1L_syn.append(Synapses(PEN_groups[0], EPG_groups[15], Ach_eqs_PE1L, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1L_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG8-15
    for k4 in range(0,7):
        # PEN to EPG
        PE2L_syn2.append(Synapses(PEN_groups[k4+9], EPG_groups[k4+8], Ach_eqs_PE2L2, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2L_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1L_syn2.append(Synapses(PEN_groups[k4+10], EPG_groups[k4+8], Ach_eqs_PE1L2, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1L_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE1L_syn2.append(Synapses(PEN_groups[9], EPG_groups[15], Ach_eqs_PE1L2, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1L_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)


    print('vertical connections')
    #########################
    ## vertical connections
    #########################
    
    # EPG_EPG
    for k in range(0,16):
        # EPG to EPG
        EPG_1_syn.append(Synapses(EPG_1_groups[k], EPG_1_groups[k], Ach_eqs_1, on_pre='s_ach += w_EE', method='euler'))
        EPG_1_syn[k].connect(condition='i != j')
    
    # # PEN_PEN
    # for k2 in range(0,16):
    #     # PEN to PEN
    #     PEN_1_syn.append(Synapses(PEN_1_groups[k2], PEN_1_groups[k2], Ach_eqs_PP_1, on_pre='s_ach += w_PP', method='euler'))
    #     PEN_1_syn[k2].connect(condition='i != j')
    
    # EPG to R
    S_EI_1 = Synapses(EPG_1, R, model=Ach_eqs_EI_1, on_pre='s_ach += w_EI', method='euler')
    for a in range(0,48):
        for b in range(0,3):
            S_EI_1.connect(i=a, j=b)
    

    # R to EPG
    # on_pre='''s_GABAA += w_IE
    #     s_GABAA = clip(s_GABAA, 0, g_GABAA_max)'''
    S_IE_1 = Synapses(R, EPG_1, model=GABA_eqs_1, on_pre='s_GABAA += w_IE', method='euler')
    for a2 in range(0,48):
        for b2 in range(0,3):
            S_IE_1.connect(i=b2, j=a2)
    
    # # R_R with symmetric connections
    # S_II_1 = Synapses(R, R, model=GABA_eqs_i, on_pre='s_GABAA += w_II', method='euler')
    # S_II_1.connect(condition='i != j')
    
    # EPG_PEN synapse
    for k3 in range(0,16):
        # EPG to PEN
        EP_1_syn.append(Synapses(EPG_1_groups[k3], PEN_groups[k3], Ach_eqs_EP_1, on_pre='s_ach += w_EP', method='euler'))
        EP_1_syn[k3].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    ###
    
    # PEN -> EPG_1 synapse 
    # PEN0-6 -> EPG0-8
    for k4 in range(0,7):
        # PEN to EPG
        PE2R_1_syn.append(Synapses(PEN_groups[k4], EPG_1_groups[k4+1], Ach_eqs_PE2R_1, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2R_1_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1R_1_syn.append(Synapses(PEN_groups[k4], EPG_1_groups[k4+2], Ach_eqs_PE1R_1, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1R_1_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE1R_1_syn.append(Synapses(PEN_groups[6], EPG_1_groups[0], Ach_eqs_PE1R_1, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1R_1_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG0-8 
    for k4 in range(0,7):
        # PEN to EPG
        PE2R_1_syn2.append(Synapses(PEN_groups[k4+9], EPG_1_groups[k4+1], Ach_eqs_PE2R2_1, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2R_1_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1R_1_syn2.append(Synapses(PEN_groups[k4+9], EPG_1_groups[k4+2], Ach_eqs_PE1R2_1, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1R_1_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE1R_1_syn2.append(Synapses(PEN_groups[15], EPG_1_groups[0], Ach_eqs_PE1R2_1, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1R_1_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN7-8
    PE7_1_syn = []
    # PEN7 -> EPG connections
    PE7_1_syn.append(Synapses(PEN_groups[7], EPG_1_groups[0], Ach_eqs_PE7_1, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7_1_syn.append(Synapses(PEN_groups[7], EPG_1_groups[1], Ach_eqs_PE7_1, on_pre='s_ach += 1*w_PE', method='euler'))
    PE7_1_syn.append(Synapses(PEN_groups[7], EPG_1_groups[15], Ach_eqs_PE7_1, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7_1_syn.append(Synapses(PEN_groups[7], EPG_1_groups[14], Ach_eqs_PE7_1, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE7_1_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE8v_syn = []
    # PEN8 -> EPG connections
    PE8_1_syn.append(Synapses(PEN_groups[8], EPG_1_groups[0], Ach_eqs_PE8_1, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8_1_syn.append(Synapses(PEN_groups[8], EPG_1_groups[1], Ach_eqs_PE8_1, on_pre='s_ach += 1*w_PE', method='euler'))
    PE8_1_syn.append(Synapses(PEN_groups[8], EPG_1_groups[15], Ach_eqs_PE8_1, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8_1_syn.append(Synapses(PEN_groups[8], EPG_1_groups[14], Ach_eqs_PE8_1, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE8_1_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN0-6 -> EPG8-15
    for k4 in range(0,7):
        # PEN to EPG
        PE2L_1_syn.append(Synapses(PEN_groups[k4], EPG_1_groups[k4+8], Ach_eqs_PE2L_1, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2L_1_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1L_1_syn.append(Synapses(PEN_groups[k4+1], EPG_1_groups[k4+8], Ach_eqs_PE1L_1, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1L_1_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
        
    PE1L_1_syn.append(Synapses(PEN_groups[0], EPG_1_groups[15], Ach_eqs_PE1L_1, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1L_1_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG8-15
    for k4 in range(0,7):
        # PEN to EPG
        PE2L_1_syn2.append(Synapses(PEN_groups[k4+9], EPG_1_groups[k4+8], Ach_eqs_PE2L2_1, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2L_1_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1L_1_syn2.append(Synapses(PEN_groups[k4+10], EPG_1_groups[k4+8], Ach_eqs_PE1L2_1, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1L_1_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE1L_1_syn2.append(Synapses(PEN_groups[9], EPG_1_groups[15], Ach_eqs_PE1L2_1, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1L_1_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    
    print("All synapse connections completed!")

    # record model state
    
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

    PRM0_1 = PopulationRateMonitor(EPG_1_groups[0])
    PRM1_1 = PopulationRateMonitor(EPG_1_groups[1]) 
    PRM2_1 = PopulationRateMonitor(EPG_1_groups[2]) 
    PRM3_1 = PopulationRateMonitor(EPG_1_groups[3]) 
    PRM4_1 = PopulationRateMonitor(EPG_1_groups[4]) 
    PRM5_1 = PopulationRateMonitor(EPG_1_groups[5])
    PRM6_1 = PopulationRateMonitor(EPG_1_groups[6])
    PRM7_1 = PopulationRateMonitor(EPG_1_groups[7])
    PRM8_1 = PopulationRateMonitor(EPG_1_groups[8])
    PRM9_1 = PopulationRateMonitor(EPG_1_groups[9])
    PRM10_1 = PopulationRateMonitor(EPG_1_groups[10])
    PRM11_1 = PopulationRateMonitor(EPG_1_groups[11])
    PRM12_1 = PopulationRateMonitor(EPG_1_groups[12])
    PRM13_1 = PopulationRateMonitor(EPG_1_groups[13])
    PRM14_1 = PopulationRateMonitor(EPG_1_groups[14])
    PRM15_1 = PopulationRateMonitor(EPG_1_groups[15])

    PRM0p = PopulationRateMonitor(PEN_groups[0])
    PRM1p = PopulationRateMonitor(PEN_groups[1])
    PRM2p = PopulationRateMonitor(PEN_groups[2])
    PRM3p = PopulationRateMonitor(PEN_groups[3])
    PRM4p = PopulationRateMonitor(PEN_groups[4])
    PRM5p = PopulationRateMonitor(PEN_groups[5])
    PRM6p = PopulationRateMonitor(PEN_groups[6])
    PRM7p = PopulationRateMonitor(PEN_groups[7])
    PRM8p = PopulationRateMonitor(PEN_groups[8])
    PRM9p = PopulationRateMonitor(PEN_groups[9])
    PRM10p = PopulationRateMonitor(PEN_groups[10])
    PRM11p = PopulationRateMonitor(PEN_groups[11])
    PRM12p = PopulationRateMonitor(PEN_groups[12])
    PRM13p = PopulationRateMonitor(PEN_groups[13])
    PRM14p = PopulationRateMonitor(PEN_groups[14])
    PRM15p = PopulationRateMonitor(PEN_groups[15])
    
    # PRM0p_1 = PopulationRateMonitor(PEN_1_groups[0])
    # PRM1p_1 = PopulationRateMonitor(PEN_1_groups[1])
    # PRM2p_1 = PopulationRateMonitor(PEN_1_groups[2])
    # PRM3p_1 = PopulationRateMonitor(PEN_1_groups[3])
    # PRM4p_1 = PopulationRateMonitor(PEN_1_groups[4])
    # PRM5p_1 = PopulationRateMonitor(PEN_1_groups[5])
    # PRM6p_1 = PopulationRateMonitor(PEN_1_groups[6])
    # PRM7p_1 = PopulationRateMonitor(PEN_1_groups[7])
    # PRM8p_1 = PopulationRateMonitor(PEN_1_groups[8])
    # PRM9p_1 = PopulationRateMonitor(PEN_1_groups[9])
    # PRM10p_1 = PopulationRateMonitor(PEN_1_groups[10])
    # PRM11p_1 = PopulationRateMonitor(PEN_1_groups[11])
    # PRM12p_1 = PopulationRateMonitor(PEN_1_groups[12])
    # PRM13p_1 = PopulationRateMonitor(PEN_1_groups[13])
    # PRM14p_1 = PopulationRateMonitor(PEN_1_groups[14])
    # PRM15p_1 = PopulationRateMonitor(PEN_1_groups[15])

    PRMR = PopulationRateMonitor(R)
    mon_R = StateMonitor(R, ['v', 'IsynEI', 'IsynEI_1', 'Isyn_ii', 'IsynEItot',], record=True)
    mon_EPG = StateMonitor(EPG, ['v', 'Isyn', 'Isyn_i', 'Isyn_PE'], record=True)
    mon_EPG_1 = StateMonitor(EPG_1, ['v', 'Isyn_1', 'Isyn_i_1', 'Isyn_PE_1'], record=True)
    mon_syn = StateMonitor(S_IE, ['s_GABAA', 'Isyn_i_post'], record=True)
    mon_syn_1 = StateMonitor(S_IE_1, ['s_GABAA', 'Isyn_i_1_post'], record=True)

    # SM = SpikeMonitor(EPG)
    # SM_1 = SpikeMonitor(EPG_1)
    print('collect')
    net=Network(collect())
    net.add(EPG_groups,EPG_syn,PEN_groups,PEN_syn,EP_syn,PE2R_syn,PE2L_syn,PE1R_syn,PE1L_syn,PE2R_syn2,PE2L_syn2,PE1R_syn2,PE1L_syn2,PE7_syn,PE8_syn)
    net.add(EPG_1_groups,EPG_1_syn,EP_1_syn,PE2R_1_syn,PE2L_1_syn,PE1R_1_syn,PE1L_1_syn,PE2R_1_syn2,PE2L_1_syn2,PE1R_1_syn2,PE1L_1_syn2,PE7_1_syn,PE8_1_syn)

    # run simulation

    ## SIMULATION ###
    print('visual cue')
    stimulus_location %= 2*np.pi
    theta_r = stimulus_location/2
    theta_l = theta_r + np.pi
    
    stimulus_location_1 %= 2*np.pi
    theta_r_1 = stimulus_location_1/2
    theta_l_1 = theta_r_1 + np.pi
    
    A = stimulus_strength
    A_1 = stimulus_strength_1
    
    for i in range(0,8):
        EPG_groups[i].I = visual_cue(theta_r, i, A)
    for i in range(8,16):
        EPG_groups[i].I = visual_cue(theta_l, i, A)
        
    for i in range(0,8):
        EPG_1_groups[i].I = visual_cue(theta_r_1, i, A_1)
    for i in range(8,16):
        EPG_1_groups[i].I = visual_cue(theta_l_1, i, A_1)
    
    net.run(t_epg_open*ms)

    for i in range(0,16):
        EPG_groups[i].I = 0
    for i in range(0,16):
        EPG_1_groups[i].I = 0
    net.run(t_epg_close * ms)

    print('body rotation')
    if half_PEN == 'right':
        for i in range(8): PEN_groups[i].I = shifter_strength
    elif half_PEN == 'left':
        for i in range(8,16): PEN_groups[i].I = shifter_strength
    # if half_PEN_1 == 'right':
    #     for i in range(8): PEN_1_groups[i].I = shifter_strength_1
    # elif half_PEN_1 == 'left':
    #     for i in range(8,16): PEN_1_groups[i].I = shifter_strength_1
            
    net.run(t_pen_open * ms)
    end  = time.time()
    print(f'\r{time.strftime("%H:%M:%S")} : {(end - start)//60:.0f} min {(end - start)%60:.1f} sec -> eval end', flush=True)
    
    device.build(run=True, clean=True)

    fr = [PRM0.smooth_rate(width=5*ms),
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
    
    fr_1 = [PRM0_1.smooth_rate(width=5*ms),
                    PRM1_1.smooth_rate(width=5*ms),
                    PRM2_1.smooth_rate(width=5*ms),
                    PRM3_1.smooth_rate(width=5*ms),
                    PRM4_1.smooth_rate(width=5*ms),
                    PRM5_1.smooth_rate(width=5*ms),
                    PRM6_1.smooth_rate(width=5*ms),
                    PRM7_1.smooth_rate(width=5*ms),
                    PRM8_1.smooth_rate(width=5*ms),
                    PRM9_1.smooth_rate(width=5*ms),
                    PRM10_1.smooth_rate(width=5*ms),
                    PRM11_1.smooth_rate(width=5*ms),
                    PRM12_1.smooth_rate(width=5*ms),
                    PRM13_1.smooth_rate(width=5*ms),
                    PRM14_1.smooth_rate(width=5*ms),
                    PRM15_1.smooth_rate(width=5*ms),]
    
    fr_pen = [PRM0p.smooth_rate(width=5*ms),
                       PRM1p.smooth_rate(width=5*ms),
                       PRM2p.smooth_rate(width=5*ms),
                       PRM3p.smooth_rate(width=5*ms),
                       PRM4p.smooth_rate(width=5*ms),
                       PRM5p.smooth_rate(width=5*ms),
                       PRM6p.smooth_rate(width=5*ms),
                       PRM7p.smooth_rate(width=5*ms),
                       PRM8p.smooth_rate(width=5*ms),
                       PRM9p.smooth_rate(width=5*ms),
                       PRM10p.smooth_rate(width=5*ms),
                       PRM11p.smooth_rate(width=5*ms),
                       PRM12p.smooth_rate(width=5*ms),
                       PRM13p.smooth_rate(width=5*ms),
                       PRM14p.smooth_rate(width=5*ms),
                       PRM15p.smooth_rate(width=5*ms),]
    
    # fr_pen_1 = [PRM0p_1.smooth_rate(width=5*ms),
    #                    PRM1p_1.smooth_rate(width=5*ms),
    #                    PRM2p_1.smooth_rate(width=5*ms),
    #                    PRM3p_1.smooth_rate(width=5*ms),
    #                    PRM4p_1.smooth_rate(width=5*ms),
    #                    PRM5p_1.smooth_rate(width=5*ms),
    #                    PRM6p_1.smooth_rate(width=5*ms),
    #                    PRM7p_1.smooth_rate(width=5*ms),  
    #                    PRM8p_1.smooth_rate(width=5*ms),
    #                    PRM9p_1.smooth_rate(width=5*ms),
    #                    PRM10p_1.smooth_rate(width=5*ms),
    #                    PRM11p_1.smooth_rate(width=5*ms),
    #                    PRM12p_1.smooth_rate(width=5*ms),
    #                    PRM13p_1.smooth_rate(width=5*ms),
    #                    PRM14p_1.smooth_rate(width=5*ms),
    #                    PRM15p_1.smooth_rate(width=5*ms),]
    
    fr_r = [PRMR.smooth_rate(width=5*ms),]
    
    fr_pen = np.array(fr_pen)
    # fr_pen_1 = np.array(fr_pen_1)
    fr_r = np.array(fr_r)

    fr = np.array(fr)
    fr_1 = np.array(fr_1)
    t = np.linspace(0, len(fr[0])/10000, len(fr[0]))
    return t, fr, fr_1, fr_pen, fr_r, mon_R, mon_syn, mon_syn_1, mon_EPG, mon_EPG_1

if __name__ == '__main__':
    t, fr, fr_1, fr_pen, fr_r, mon_R, mon_syn, mon_syn_1, mon_EPG, mon_EPG_1 = simulator()    