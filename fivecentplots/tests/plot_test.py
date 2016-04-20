import sys
sys.path.append(r'C:\GitHub\fivecentplots')
import fivecentplots as fcp
import pandas as pd

df = pd.read_csv(r'fake_data.csv')
df_box = pd.read_csv(r'fake_data_box.csv')

# Single IV curve grouped by die
sub = df[(df.Substrate=='Si') &
         (df['Target Wavelength']==450) &
         (df['Boost Level']==0.2) &
         (df['Temperature [C]']==25).copy()
        ]
d = fcp.plot(df=sub, x='Voltage', y='I [A]', leg_groups='Die', show=True)

# Facet grid by boost level and temperature
sub = df[(df.Substrate=='Si') &
          (df['Target Wavelength']==450)].copy()
d = fcp.plot(df=sub, x='Voltage', y='I [A]', leg_groups='Die',
             row='Boost Level', col='Temperature [C]', xticks=4, show=True)

# Facet grid by boost level and temperature (no axis sharing)
sub = df[(df.Substrate=='Si') &
         (df['Target Wavelength']==450)].copy()
d = fcp.plot(df=sub, x='Voltage', y='I [A]', leg_groups='Die',
             row='Boost Level', col='Temperature [C]',
             sharex=False, sharey=False, ax_size=[200,300], show=True)

# Facet grid by boost level and temperature with transformation
sub = df[(df.Substrate=='Si') &
         (df['Target Wavelength']==450)].copy()
d = fcp.plot(df=sub, x='Voltage', y='I [A]', leg_groups='Die',
             row='Boost Level', col='Temperature [C]', ax_scale='logx',
             ytrans=('pow',4), ymin=1E-8, ymax=1E-2, show=True)  #issues here with ranges, ticks

# Multiple y on same axis with filter
filt = 'Substrate=="Si" & Target_Wavelength==450 & Boost_Level==0.2 & ' \
       'Temperature_C==25'
d = fcp.plot(df=df, x='Voltage', y=['I [A]', 'Voltage'], filter=filt,
             leg_groups='Die', ylabel='Values', show=True)

# Multiple y on same axis with filter and twinx
filt = 'Substrate=="Si" & Target_Wavelength==450 & Boost_Level==0.2 & ' \
       'Temperature_C==25'
d = fcp.plot(df=df, x='Voltage', y=['I [A]', 'Voltage'], filter=filt,
             show=True, leg_groups='Die', ylabel='I [A]', ylabel2='Voltage')

# Boxplot test
d = fcp.boxplot(df=df_box, y='Value', groups=['Batch', 'Sample'], show=True)