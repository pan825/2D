from equations_1Rv import *


EPG0_eqs = EPG_equation(0)
EPG1_eqs = EPG_equation(1)


connection_types = ['PE2R', 'PE2L', 'PE1R', 'PE1L', 'PE2R2', 'PE2L2', 
                       'PE1R2', 'PE1L2', 'PE7', 'PE8',]
    
ach_equations = {}
for i in range(0,2):
    for conn_type in connection_types:
        target_var = f'Isyn_{conn_type}' if conn_type != 'EI' else 'IsynEI'
        ach_equations[f'Ach_eqs_{conn_type}_{i}'] = ach_equation(target_var, i)

# 打印查看結果
for name, eq in ach_equations.items():
    print(f"{name}:")
    print(eq)
    print("-" * 50) 