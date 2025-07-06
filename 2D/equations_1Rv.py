# model equations


eqs_R = '''
dv/dt = (IsynEItot + Isyn_ii + I + E_l - v)/taum + sigma*sqrt(2/taum)*xi : 1 (unless refractory)
I : 1
IsynEI : 1
IsynEI_1 : 1
IsynEItot = IsynEI + IsynEI_1 : 1
Isyn_ii: 1
'''

Imax  = 0.8   # pA，最終飽和值
Ihalf = 0.4  # 半飽和點
k     = 0.08

# EPG -> R
Ach_eqs_EI = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynEI_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# R -> EPG
GABA_eqs = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_i_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

# R <-> R
GABA_eqs_i = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_ii_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

# EPG_1 -> R
Ach_eqs_EI_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynEI_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# R -> EPG_1
GABA_eqs_1 = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_i_1_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

GABA_eqs_i_1 = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_ii_1_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

eqs_EPG = '''
dv/dt = ( Isyn + Isyn_i + Isyn_PE + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PE2R :1
Isyn_PE2L :1
Isyn_PE1R :1
Isyn_PE1L :1
Isyn_PE2R2:1
Isyn_PE2L2 :1
Isyn_PE1R2 :1
Isyn_PE1L2:1
Isyn_PE7:1
Isyn_PE8:1
Isyn_PE = Isyn_PE2R + Isyn_PE2L + Isyn_PE1R + Isyn_PE1L + Isyn_PE2R2 + Isyn_PE2L2 + Isyn_PE1R2 + Isyn_PE1L2 + Isyn_PE7 + Isyn_PE8:1
'''

eqs_EPG_1 = '''
dv/dt = ( Isyn_1 + Isyn_i_1 +Isyn_PE_1 + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn_1 : 1
Isyn_i_1 : 1
Isyn_PE2R_1 :1
Isyn_PE2L_1 :1
Isyn_PE1R_1 :1
Isyn_PE1L_1 :1
Isyn_PE2R2_1:1
Isyn_PE2L2_1 :1
Isyn_PE1R2_1 :1
Isyn_PE1L2_1:1
Isyn_PE7_1:1
Isyn_PE8_1:1
Isyn_PE_1 = Isyn_PE2R_1 + Isyn_PE2L_1 + Isyn_PE1R_1 + Isyn_PE1L_1 + Isyn_PE2R2_1 + Isyn_PE2L2_1 + Isyn_PE1R2_1 + Isyn_PE1L2_1 + Isyn_PE7_1 + Isyn_PE8_1:1
'''

eqs_PEN = '''
dv/dt = (Isyn_pp + Isyn_EP + Isyn_EP_1 + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn_pp : 1
Isyn_EP : 1
Isyn_EP_1 : 1
Isyn_EP_tot = Isyn_EP + Isyn_EP_1 : 1
'''

# EPG <-> EPG
Ach_eqs = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# PEN <-> PEN
Ach_eqs_PP = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_pp_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_EP = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_EP_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2R = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2R_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2L = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2L_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1L = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1L_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1R = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1R_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE2R2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2R2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2L2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2L2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1L2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1L2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1R2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1R2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE7 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE7_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE8 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE8_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''



#########

# model equations

Ach_eqs_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PP_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_pp_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_EP_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_EP_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2R_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2R_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2L_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2L_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1L_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1L_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1R_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1R_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE2R2_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2R2_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2L2_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2L2_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1L2_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1L2_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1R2_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1R2_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE7_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE7_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE8_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE8_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
