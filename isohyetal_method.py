import pandas as pd
import numpy as np

def isohyetal_method(df : pd.DataFrame = None):
    '''
    This function is used to calculate the isohyetal method
    '''
    # create a new dataframe
    df_new = pd.DataFrame()
    # create a new column if the column names are exist
    if 'Area' in df.columns: df_new['Area'] = df['Area']
    if 'Precipitation' in df.columns: df_new['Precipitation'] = df['Precipitation']

    # if the column names "Area" and "Precipitation" are not exist, then return the error message
    assert 'Area' in df_new.columns, 'The column name "Area" is not exist'
    assert 'Precipitation' in df_new.columns, 'The column name "Precipitation" is not exist'

    # calculate by using the isohyetal method
    return (df_new['Area'] * df_new['Precipitation']).sum() / df_new['Area'].sum()

# make sample of dataframe and call the function
dict_sampled = {'Area': [1, 2, 3, 4, 5], 'Precipitation': [1, 2, 3, 4, 5]}
df = pd.DataFrame(dict_sampled,index = ['A', 'B', 'C', 'D', 'E'])

# use exmaple
# precipitation = isohyetal_method(df)
# print(precipitation.round(2), 'mm', sep='')