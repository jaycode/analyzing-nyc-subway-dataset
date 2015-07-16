from pandas import *
from ggplot import *

df = pandas.read_csv('turnstile_weather_v2.csv')

entriesByDayOfMonth = df[['DATEn', 'ENTRIESn_hourly']] \
    .groupby('DATEn', as_index=False).mean()

entriesByDayOfMonth['Day'] = [datetime.strptime(x, '%m-%d-%y').strftime('%w %a') \
     for x in entriesByDayOfMonth['DATEn']]

entriesByDay = entriesByDayOfMonth[['Day', 'ENTRIESn_hourly']]. \
    groupby('Day', as_index=False).mean()

print df
print entriesByDay

plot = ggplot(entriesByDay, aes(x='Day', y='ENTRIESn_hourly')) + \
    geom_histogram(aes(weight='ENTRIESn_hourly')) + \
    ggtitle('NYC Subway ridership by day of week') + xlab('Day') + ylab('Average Entries per hour')

print plot