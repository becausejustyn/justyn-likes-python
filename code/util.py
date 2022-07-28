
import pandas as pd

def format_decimals_factory(num_decimals = 1):
    return lambda x: "{1:.{0}f}".format(num_decimals, x)

def map_df(min: int, max: int, variable):
  data = pd.concat([nfl.load_pbp_data(variable).assign(variable = variable) for variable in range(min, max)])
  
