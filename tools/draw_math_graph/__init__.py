print("Initializing draw_math")
from .draw_x_math import draw_coordinate_system,\
draw_liner_function,\
draw_quadratic_function,\
draw_power_function,\
draw_exponential_function,\
draw_log_function

from .draw_xy_math import draw_3D_expression,draw_decision_boundray

# 可以定义包级别的变量
__all__ = ['draw_coordinate_system', 'draw_liner_function',
           'draw_quadratic_function','draw_power_function',
           'draw_exponential_function','draw_log_function',
           'draw_3D_expression','draw_decision_boundray']
