from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric

# fill missing values with a value at the same time one day ago
def fill_missing(values):
	previous_day_ref_distance = 1
	for row in range(values.shape[0]):
		for col in range(1):
			v = values[row, col]
			if isnan(v) or v <= 1:
				values[row, col] = values[row - previous_day_ref_distance, col]


# load all data
dataset = read_csv('../resources/electricity_consumption_orginal.csv', sep=',', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
# mark all missing values
dataset.replace('?', nan, inplace=True)
# make dataset numeric
dataset = dataset.astype('float32')

#Need to delete 24:00:00 datas
dataset.to_csv('../resources/electricity_consumption_30min.csv')



# resample minute data to total for each 30 minutes
dataset = read_csv('../resources/electricity_consumption_30min.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
thirty_minute_groups = dataset.resample('30min')
thirty_minute_data = thirty_minute_groups.mean()
fill_missing(thirty_minute_data.values)
print(thirty_minute_data.shape)
print(thirty_minute_data.head())
thirty_minute_data.to_csv('../resources/electricity_consumption_30min.csv')


# resample minute data to total for each day
dataset = read_csv('../resources/electricity_consumption_30min.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
daily_minute_groups = dataset.resample('D')
daily_minute_data = daily_minute_groups.mean()
print(daily_minute_data.shape)
print(daily_minute_data.head())
daily_minute_data.to_csv('../resources/electricity_consumption_daily.csv')