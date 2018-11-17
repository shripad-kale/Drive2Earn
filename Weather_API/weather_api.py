
# coding: utf-8

from weather import Weather, Unit

'''
A Python wrapper for the Yahoo Weather API.

With the API, you can get up-to-date weather information for any location, including 5-day forecast, wind, atmosphere, astronomy conditions, and more. 
You can lookup weather by woeid, city name or lat/long.
'''

#Lookup via location name.
def get_weather_locaion(location):
    weather = Weather(unit=Unit.CELSIUS)
    location = weather.lookup_by_location(location)
    condition = location.condition
    print(condition.text)
	return condition.text


#Get weather forecasts for the upcoming days on city name.
def get_weather_location_week(city):
    weather = Weather(unit=Unit.CELSIUS)
    location = weather.lookup_by_location(city)
    forecasts = location.forecast
    for forecast in forecasts:
        print(forecast.text)
        print(forecast.date)
        print(forecast.high)
        print(forecast.low)
	return forecast


#Lookup via latitude and longitude
def get_weather_lat_long(latitude,longitude):           
    weather = Weather(Unit.CELSIUS)
    lookup = weather.lookup_by_latlng(latitude,longitude)
    condition = lookup.condition
    print(condition.text)
	return condition.text

def get_weather()
	weather = Weather(unit=Unit.CELSIUS)
	lookup = weather.lookup(560743)
	condition = lookup.condition
	print(condition.text)
	return condition.text
