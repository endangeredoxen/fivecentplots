import sys
sys.path.append(r'C:\Users\Steve\PycharmProjects\fivecentplots')
import fivecentplots as fcp
import pandas as pd

df = pd.read_csv(r'fake_data.csv')

# # Single IV curve grouped by die
# sub = df[(df.Substrate=='Si') &
#          (df['Target Wavelength']==450) &
#          (df['Boost Level']==0.2) &
#          (df['Temperature [C]']==25)
#         ]
# d = fcp.plot(sub, 'Voltage', 'I [A]', leg_groups='Die')

# Facet grid by boost level and temperature
sub = df[(df.Substrate=='Si') &
         (df['Target Wavelength']==450)
        ]
d = fcp.plot(sub, 'Voltage', 'I [A]', leg_groups='Die', row='Boost Level',
             col='Temperature [C]')
