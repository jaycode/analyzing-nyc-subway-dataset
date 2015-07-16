from pandas import *
from ggplot import *

df = pandas.read_csv('turnstile_weather_v2.csv')
df_rain = df[df['rain'] == 1].reset_index().drop('index', 1)
df_norain = df[df['rain'] == 0].reset_index().drop('index', 1)

plot1 = ggplot(df_rain, aes('ENTRIESn_hourly')) +\
    geom_histogram() + ggtitle("Entries hourly when raining") +\
    ylab("Frequency") + xlab("Hourly Entries") + xlim(0, 20000) + ylim(0, 20000)

plot2 = ggplot(df_norain, aes('ENTRIESn_hourly')) +\
    geom_histogram() + ggtitle("Entries hourly when not raining") +\
    ylab("Frequency") + xlab("Hourly Entries") + xlim(0, 20000) + ylim(0, 20000)

print plot1
print plot2