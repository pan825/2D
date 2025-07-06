from brian2 import *

def create_ach_equation(target_var_name):
    """
    創建 ACh 突觸方程式的函數
    
    參數:
        target_var_name: 目標變數名稱，例如 'Isyn_PE2R'
    
    返回:
        完整的方程式字串
    """
    return f'''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
{target_var_name}_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

def create_gaba_equation(target_var_name):
    """
    創建 GABA 突觸方程式的函數
    
    參數:
        target_var_name: 目標變數名稱，例如 'Isyn_i'
    
    返回:
        完整的方程式字串
    """
    return f'''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
{target_var_name}_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

# 使用示例
if __name__ == "__main__":
    # 替代原來的重複定義
    # Ach_eqs_PE2R = '''ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)...'''
    # Ach_eqs_PE2L = '''ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)...'''
    
    # 現在可以用函數動態生成
    connection_types = ['PE2R', 'PE2L', 'PE1R', 'PE1L', 'PE2R2', 'PE2L2', 
                       'PE1R2', 'PE1L2', 'PE7', 'PE8', 'EI', 'EP', 'PP']
    
    # 動態創建所有 ACh 方程式
    ach_equations = {}
    for conn_type in connection_types:
        target_var = f'Isyn_{conn_type}' if conn_type != 'EI' else 'IsynEI'
        ach_equations[f'Ach_eqs_{conn_type}'] = create_ach_equation(target_var)
    
    # 打印查看結果
    for name, eq in ach_equations.items():
        print(f"{name}:")
        print(eq)
        print("-" * 50) 