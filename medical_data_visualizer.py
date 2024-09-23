# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importing data from medical_examination.csv and assign it to df variable

df = pd.read_csv('/Users/a1/Downloads/medical_examination.csv')

# print(df.head())

# Create the overweight column in the df variable 
height_in_cm = df['height']
height_in_m = height_in_cm / 100
bmi = df['weight'] / height_in_m ** 2
df['overweight'] = bmi.apply(lambda x: 1 if x > 25 else 0)
print(df['overweight'].head())

# df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype(int) -- Another approach


# Normalize the data by making 0 always good and 1 always bad. If the value of cholestrol or gluc is 1, amke the value 0.

df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x:0 if x == 1 else 1)

# Draw the Categorical Plot in the draw_cat_plot function

def draw_cat_plot():
    df_cat = pd.melt(df, 
                    id_vars = ['cardio'], 
                    value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    
# # Group and count values
    df_cat['total'] = 1
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count().reset_index()

# # Convert the data into long format and create a chart that shows the value counts of the categrorical features using sns.catplot()
    df_cat['cardio'] = df_cat['cardio'].astype('category')
    df_cat['variable'] = df_cat['variable'].astype('category')
    df_cat['value'] = df_cat['value'].astype('category')

    fig = sns.catplot(x = 'variable', y = 'total', hue = 'value', col = 'cardio', data = df_cat, kind = 'bar').fig
    
    fig.savefig('catplot.png')
    return fig

draw_cat_plot()

def draw_heat_map():
    df_heat = df
    df_heat = df_heat[
        (df_heat['ap_lo'] <= df_heat['ap_hi']) &
        (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
        (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix and store in a corr variable

    corr = df_heat.corr()

    # Generate a mask for the upper triangle and store in the mask variable

    mask = np.triu(np.ones_like(corr, dtype= 'float64'))

    # Set the matplotlib figure 

    fig, ax = plt.subplots(figsize=(11, 9))

    #  Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap()
    
    sns.heatmap(corr, mask=mask, center=0, fmt='.1f', annot=True, vmax=0.1, linewidths=0.5, cbar_kws={"shrink": .5})

    fig.savefig('heatmap.png')
    return fig

plt.show()
