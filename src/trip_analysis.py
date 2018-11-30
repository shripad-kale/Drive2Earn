
import pandas as pd
import numpy as np
import os
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 25, 10
import matplotlib.pyplot as plt
from matplotlib import pyplot
import datetime as dt
import matplotlib.pyplot as plt
# Convert Accelerometer reading to g units, 8bit data with a range of +-2g #
def convert_acc(x):
    x = int(str(x), 16)
    if x > 127:
        x = x - 256
    return np.float64(x * 0.01536)


# Convert a single row of Accelerometer data to engineering units and return a df #
def convert_acc_row(row):
    # Initially the data was gathered without magnetometer, so check length for identification #
    data_list = []
    for i in range(0, len(row), 6):
        x = convert_acc(row[i:i + 2])
        y = convert_acc(row[i + 2:i + 4])
        z = convert_acc(row[i + 4:i + 6])

        data_list.append([x, y, z])
    return pd.DataFrame(columns=['ax', 'ay', 'az'], data=data_list)


def making_trips_alone(df,id_of_trip):
    df_trip = df[df['tripID']==id_of_trip]
    trip_ts['timestamp']=df_trip['timeStamp']
    trip_ts['speed']=df_trip['speed']
    return trip_ts

def analysis_of_trip(trip_ts,window_size):
    trip_ts.plot(color='gray')
    r=trip_ts.rolling(window=window_size,center=True)
    r.mean().plot(color='red')

def get_df_trip(df,trip_id):
    #for trip_id in set(df['tripID'].values):
    df_trip = df[df['tripID'] == trip_id]
   # print(df_trip)
    return df_trip
def speed_details(df):

    #trips = df['tripID'].unique()

    trip_ts = pd.DataFrame()
    trip_ts['timestamp'] = df_trip['timeStamp']
    trip_ts['speed'] = df_trip['speed']

#Trip time and Night Driving and Idling with Engine ON
def calculat_trip_time(df_trip):
    temp_df = df_trip.iloc[:, [2, 3, 14]]
    t = temp_df['accData']
    start = temp_df['timeStamp'].values[0]
    end = temp_df['timeStamp'].values[t.shape[0] - 1]
    start_dt = dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end_dt = dt.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    diff = (end_dt - start_dt)
    trip_time_minutes=diff.seconds / 60
    print('The trip time is for this journey in minutes {} \ntrip start time {}\ntrip end time is {}'.format(
        trip_time_minutes, start, end))
    count_daylight, count_night = 0, 0
    for timestamp in temp_df.timeStamp.values:
        time = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        hr, mi = (time.hour, time.minute)
        if hr >= 7 and hr < 18:
            count_daylight = count_daylight + 1
        else:
            count_night = count_night + 1
    trip_in_day_time=count_daylight / 60
    trip_in_night_time=count_night / 60
    trip_in_day_percentage =  round(((count_daylight / df_trip.shape[0]) * 100),2)
    trip_in_night_percentage  =round(( (count_night / df_trip.shape[0]) * 100),2)
    trip_dict={
        "trip_time_minutes":trip_time_minutes,
        "start_time":start,
        "end_time":end,
        "trip_in_day_time":trip_in_day_time,
        "trip_in_night_time":trip_in_night_time,
        "trip_in_day_percentage" : trip_in_day_percentage,
        "trip_in_night_percentage":trip_in_night_percentage,
    }
    print("travel time in DayLight in minutes: {}\ntravel time in Night or Evening in minutes:{}".format(
        trip_in_day_time,trip_in_night_time))
    print("percentage of time for trip when driving is done in daylight for whole trip: {} ".format(
        trip_in_day_percentage))
    print("percentage of time for trip when driving is done in Night for whole trip: {} ".format(
        trip_in_night_percentage))
    return trip_dict

def speed_plot(bars1):
    # Data
    size = []
    names = 'speed_0-10', 'speed_10-20', 'speed_20-30', 'speed_30-40', 'speed_40-50', 'speed_50-60', 'speed_60-70',
    for i in range(7):
        bool_series = bars1.between((i*10), (i + 1) * 10, inclusive=True)
        print(i)
        size.append(bars1[bool_series].count())

    print(size)
    # create a figure and set different background
    fig = plt.figure(figsize=(5, 5))
    fig.patch.set_facecolor('black')

    # Change color of text
    plt.rcParams['text.color'] = 'white'

    # Create a circle for the center of the plot
    my_circle = plt.Circle((0, 0), 0.6, color='black')

    # Pieplot + circle on it
    plt.pie(size, labels=names)
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()
    plt.savefig('img\speed_details' + '.png')



#get overspeed percentage time in minutes of trip with threshhotl 60
def trip_speed_analysis(df_trip):
    threshold=60
    trip_ts = pd.DataFrame()
    trip_ts['timestamp'] = df_trip['timeStamp']
    trip_ts['speed'] = df_trip['speed']
    temp = trip_ts.iloc[:, [0, 1]]
    temp['timestamp'] = pd.to_datetime(temp['timestamp'])
    temp = temp.set_index(['timestamp'])
    bars1 = temp.speed.resample('min', how='max')
    over_speed_minute=round(bars1[bars1.values>threshold].count(),4)
    print(over_speed_minute)
    over_speed_percentage = round((over_speed_minute / (bars1.size))*100,2)
    print("percentage of time intervals the drivers has overspeed over time intervals of a minute :",
          (over_speed_percentage))

    speed_plot(bars1)
    return over_speed_percentage

# creating alert for driver for which time period in battery level is low
def low_bettery_level_alert(df_trip):
    threshold_min_value_for_battery = 13.00
    count_battery_alert = 0
    for i in df_trip.battery.values:
        if i < threshold_min_value_for_battery:
            count_battery_alert = count_battery_alert + 1
        low_bettery_time=count_battery_alert / 60
    low_bettery_percentage= round((count_battery_alert / df_trip.shape[0]) * 100,2)
    print("time for trip when battery value is low  in minutes: {} ".format(low_bettery_time))
    print("percentage of time for trip when battery value is low for whole trip: {} ".format(
        low_bettery_percentage ))
    bettery_dict={"low_bettery_time":low_bettery_time ,
                   "low_bettery_percentage":low_bettery_percentage }
    return bettery_dict

## get max Acceleration for each minute time interval to know Hard acceleration/Rash driving of journey
def acceleration_dat(df_trip):
    df_acc_data = pd.DataFrame(
        data={'a_x': 1.5 * np.random.randn(df_trip.shape[0]), 'a_y': 1.1 * np.random.randn(df_trip.shape[0]),
              'a_z': 1.2 * np.random.randn(df_trip.shape[0])})

    trip_ts = pd.DataFrame()
    trip_ts['timestamp'] = df_trip['timeStamp']
    trip_ts['speed'] = df_trip['speed']
    df_acc_data['timestamp'] = trip_ts['timestamp']
    df_acc_data['ticks'] = range(0, len(df_acc_data.index.values))
    df_acc_data['Rolling_Mean_x'] = df_acc_data['a_x'].rolling(window=60).mean()
    df_acc_data['Rolling_Mean_y'] = df_acc_data['a_y'].rolling(window=60).mean()
    df_acc_data['Rolling_Mean_z'] = df_acc_data['a_z'].rolling(window=60).mean()
    df_acc_data = df_acc_data.reset_index(drop=True)
    trip_ts = trip_ts.reset_index(drop=True)
    df_acc_data.timestamp = trip_ts.timestamp
    df_acc_data = df_acc_data.set_index(['timestamp'])
    df_acc_data.index = pd.to_datetime(df_acc_data.index, unit='ns')
    acc_bars = df_acc_data.a_z.resample('min').max()
    threshold = 3.4
    count1 = acc_bars[acc_bars.values > threshold].count()
    percentage_rush_driving = round((count1 / (acc_bars.size))*100,2)
    print("percentage of time the drivers has hard brake/hard acceleration :",
          percentage_rush_driving)

    lane_change_bars = df_acc_data.a_y.resample('min').max()
    #threshold = 3.6
    count = lane_change_bars[lane_change_bars.values > threshold].count()

    percentage_lane_change = round((count / (lane_change_bars.size))*100,2)
    #print(        "percentage of time in the drivers has hard brake/hard acceleration in lane changes over time :",        percentage_lane_change)
    dict_acc={
        "percentage_lane_change":percentage_lane_change,
        "percentage_rush_driving":percentage_rush_driving

    }
    #print(df_acc_data[61:66])
    return percentage_rush_driving


df=pd.read_csv("1.0.csv")
df_trip=get_df_trip(df,11)

df_trip['accData'] = df_trip['accData'].apply(convert_acc_row)
calculat_trip_time(df_trip)
trip_speed_analysis(df_trip)
low_bettery_level_alert(df_trip)
#speed_details(df)
acceleration_dat(df_trip)