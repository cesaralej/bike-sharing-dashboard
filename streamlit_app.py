import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import requests
import time

st.set_page_config(page_title='BiciWash BI Dashboard', page_icon='🚴', layout='wide')

# Local variables
column_names = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
min_temp = -10.0
max_temp = 40.0
default_temp = 15.0
max_windspeed = 67
default_windspeed = 10

# Load the dataset
df = pd.read_csv('data/bike-sharing_hourly.csv')

# Load the model
with open('models/reg.pickle', 'rb') as file:
    model = pickle.load(file)

# Function to visualize the bike counts for a selected month and compare it to the previous year
def get_monthly_graph(dataframe, selected_month):
    dataframe['dteday'] = pd.to_datetime(dataframe['dteday'])
    daily_counts = df.groupby(['dteday'])['cnt'].sum().reset_index()
    monthly_y1_df = daily_counts[daily_counts['dteday'].dt.month == selected_month][daily_counts['dteday'].dt.year == 2011].reset_index(drop=True)
    monthly_y2_df = daily_counts[daily_counts['dteday'].dt.month == selected_month][daily_counts['dteday'].dt.year == 2012].reset_index(drop=True)
    monthly_y2_df['2011'] = monthly_y1_df['cnt']
    monthly_y2_df['2012'] = monthly_y2_df['cnt']
    compare_graph = px.line(monthly_y2_df, x='dteday', y=['2012', '2011'],
                labels={'cnt': 'Total Bike Counts', 'dteday': 'Date', 'year': 'Year'},
                title='Monthly Bike Counts VS Previous Year',
                template='ggplot2')
    return compare_graph

def get_yearly_graph(dataframe):
    dataframe['dteday'] = pd.to_datetime(dataframe['dteday'])
    daily_counts = df.groupby(['dteday'])['cnt'].sum().reset_index()
    monthly_y1_df = daily_counts[daily_counts['dteday'].dt.year == 2011].reset_index(drop=True)
    monthly_y2_df = daily_counts[daily_counts['dteday'].dt.year == 2012].reset_index(drop=True)
    monthly_y2_df['2011'] = monthly_y1_df['cnt']
    monthly_y2_df['2012'] = monthly_y2_df['cnt']
    compare_graph = px.line(monthly_y2_df, x='dteday', y=['2012', '2011'],
                labels={'cnt': 'Total Bike Counts', 'dteday': 'Date', 'year': 'Year'},
                title='Bike Counts VS Previous Year',
                template='ggplot2')
    return compare_graph

def get_yearly_avg_temp_graph(dataframe):
    dataframe['dteday'] = pd.to_datetime(dataframe['dteday'])
    daily_counts = dataframe.groupby(['dteday'])['cnt'].sum().reset_index()
    avg_temp = dataframe.groupby(['dteday'])['temp'].mean().reset_index()
    monthly_y2_df = daily_counts[daily_counts['dteday'].dt.year == 2012].reset_index(drop=True)
    monthly_y2_df['temp'] = avg_temp['temp'].apply(lambda x: x * daily_counts['cnt'].mean()*2)
    monthly_y2_df['2012'] = monthly_y2_df['cnt']
    compare_graph = px.line(monthly_y2_df, x='dteday', y=['2012', 'temp'],
                            labels={'2012': 'Total Bike Counts', 'temp': 'Temperature', 'dteday': 'Date'},
                            title='Bike Counts VS Temperature (2012)',
                            template='ggplot2')
    return compare_graph

# Function to prepare API data for prediction
def prepare_api_dataframe(response, col_names):
    weather_df = pd.DataFrame(response.json()['hourly'])
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df['mnth'] = weather_df['time'].dt.month
    weather_df['hr'] = weather_df['time'].dt.hour
    weather_df['temp'] = (weather_df['temperature_80m'] - min_temp) / (max_temp - min_temp)
    weather_df['atemp'] = (weather_df['apparent_temperature'] - min_temp) / (max_temp - min_temp)
    weather_df['windspeed'] = (weather_df['wind_speed_10m']) / (max_windspeed)
    weather_df['hum'] = weather_df['relative_humidity_2m'] / 100
    weather_df['weekday'] = weather_df['time'].dt.dayofweek
    weather_df['season'] = (weather_df['mnth'] % 12 + 3) // 3
    weather_df['holiday'] = 0 # Assume no holiday
    weather_df['workingday'] = weather_df['weekday'].apply(lambda x: 1 if x < 5 else 0)
    weather_df['weathersit'] = 1 # Assume clear weather
    weather_prepared_df = weather_df.drop(columns=['apparent_temperature', 'temperature_80m', 'wind_speed_10m', 'relative_humidity_2m', 'rain', 'snowfall'])
    weather_prepared_df = weather_prepared_df[col_names]
    return weather_df

# Function to fetch weather information from an API
def weather_request(col_names):
    url = "https://api.open-meteo.com/v1/forecast?latitude=42.361145&longitude=-71.057083&hourly=relative_humidity_2m,apparent_temperature,rain,snowfall,wind_speed_10m,temperature_80m&timezone=America%2FNew_York"

    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            #tab3.success('Request successfully sent!')
            return prepare_api_dataframe(response, col_names)
        else:
            st.error(f'Error sending request. Status Code: {response.status_code}')
    
    except Exception as e:
        st.error(f'An error occurred: {e}')

# Header components
def header():
    col1, col2 = st.columns([1, 5])
    col1.image('images/bike-sharing.jpg', use_column_width=True)
    col2.title('BiciWash BI Dashboard')
    col2.write("Welcome to BiciWash's analysis and prediction dashboard.")
    col2.write("BiciWash is an eBike sharing company that operates in Boston, MA. Data from 2011 and 2012 has been used for this dashboard.")

def get_month_max_rentals(df, year):
    if year == 2011:
        year_value = 0
    else:
        year_value = 1
    monthly_rentals = df[df['yr'] == year_value].groupby(['mnth'])['cnt'].sum()
    max_monthly_rentals = monthly_rentals.idxmax()
    max_monthly_rentals_count = monthly_rentals.max()
    max_monthly_rentals_count = "{:,.0f}".format(max_monthly_rentals_count)

    dict_month = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    result = f"The month with the most rentals was {dict_month[max_monthly_rentals]} with {max_monthly_rentals_count} rentals"
    return result

# Tab 1 content
def results_content(tab):
    tab.header('2012 Overview')
    tab.error('Here we are missing to show some other insights and visualizations')
    tab.write("Let's take a look at the bike-sharing data for 2012.")

    # month with most rentals
    
    
    tab.write(get_month_max_rentals(df, 2012))
    tab.plotly_chart(get_yearly_graph(df))
    tab.plotly_chart(get_yearly_avg_temp_graph(df))


    tab.subheader('Bike Counts by Month vs Previous Year')

    # choose a month
    month = tab.slider('Select Month', min_value=1, max_value=12, value=12)

    # Filter the data for a month
    monthly_graph = get_monthly_graph(df, month)


    # Create a Plotly line chart with both years overlaid
    tab.plotly_chart(monthly_graph)

# Tab 2 content
def insights_content(tab):
    tab.header('Descriptive Analytics')
    tab.error('Here we can talk about how we dealt with outliers and if we added new features. We can also compare our predcitions with the real values')

    # Display insights or visualizations based on the selected filters
    sel_season = tab.selectbox('Season', df['season'].unique())
    filtered_data = df[df['season'] == sel_season]
    hourly_counts = filtered_data.groupby('hr')['cnt'].mean()
    tab.line_chart(hourly_counts)
    tab.write(f'Average Bike Counts by Hour for Season {sel_season}')

def revenue_prediction_content(tab, col_names):
    # Predictive Analytics Section
    tab.header('Predictive Analytics')
    tab.error('This one is pretty much done, how can we expand on the markieting campaign? maybe a calculator to show exactly how much to spend so we can get a positive return?')
    tab.write("Grabbing the Boston weather forecast from https://open-meteo.com open-source API, we are able to predict the bike counts for the next 7 days and our earnings using a machine learning model.")
    tab.write("If the earnings are negative, the app will suggest investing in a marketing campaign.")
    tab.subheader('Revenue and Operational Cost')
    Revenue = tab.slider('Revenue per Bike', min_value=0, max_value=10, value=5)
    OperationalCost = tab.slider('Operational Cost per Hour', min_value=10, max_value=50, value=15)

    # Send request to the weather API and make predictions
    if tab.button('Press to Predict Earnings'):
        with st.spinner('Wait for it...'):
            time.sleep(3)
            forecast = weather_request(col_names)
            filtered_forecast = forecast.drop(columns=['time'])
            prediction_w1 = model.predict(filtered_forecast)
            #filtered_forecast['weathersit'] = filtered_forecast['weathersit'].apply(lambda x: 2 if x == 1 else 1)
            #prediction_w2 = model.predict(filtered_forecast)
            #filtered_forecast['weathersit'] = filtered_forecast['weathersit'].apply(lambda x: 3 if x == 1 else 1)
            #prediction_w3 = model.predict(filtered_forecast)
            #filtered_forecast['weathersit'] = filtered_forecast['weathersit'].apply(lambda x: 4 if x == 1 else 1)
            #prediction_w4 = model.predict(filtered_forecast)
            complete_df = pd.concat([forecast, pd.DataFrame(prediction_w1, columns=['predicted_cnt'])], axis=1)
            #complete_df['predicted_cnt_w2'] = prediction_w2
            #complete_df['predicted_cnt_w3'] = prediction_w3
            #complete_df['predicted_cnt_w4'] = prediction_w4
            complete_df['earnings'] = (complete_df['predicted_cnt'] * Revenue) - OperationalCost
            total_earnings = complete_df['earnings'].sum()
            total_earnings_f = "{:,.2f}".format(total_earnings)


            if total_earnings < 0:
                tab.metric('Total Earnings', f'${total_earnings_f}', f'-${total_earnings_f[1:]}')
                tab.warning('We are not expecting to make any money in the next 7 days. We should consider investing in marketing materials.')
                st.snow()
            else:
                tab.metric('Total Earnings', f'${total_earnings_f}', f'${total_earnings_f}')
                tab.success('We are expecting to make money in the next 7 days. Keep up the good work!')
                st.balloons()

            tab.subheader('7-Day Earnings Forecast')
            tab.line_chart(complete_df[['time', 'earnings']].set_index('time'))

            #compare it to the last 7 days of the dataset
            last_7_days = df.tail(168)
            last_7_days['time'] = pd.to_datetime(last_7_days['dteday'])
            last_7_days = last_7_days[['time', 'cnt']]
            last_7_days = last_7_days.set_index('time')
            tab.line_chart(last_7_days)

            # Plot the forecasted bike counts
            tab.subheader('Forecasted Bike Counts')
            tab.line_chart(complete_df[['time', 'predicted_cnt']].set_index('time'))
            #tab.line_chart(complete_df[['time', 'predicted_cnt', 'predicted_cnt_w2', 'predicted_cnt_w3', 'predicted_cnt_w4']].set_index('time'))

def simulator_content(tab, model, col_names):
    tab.header('Simulator')
    tab.error('Finish the input elements')
    tab.write("We can use this section to simulate different scenarios and see how much we can earn.")
    
    tab.header('Prediction Input')
    col1, col2 = tab.columns(2)

    seasons = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}
    months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    selected_season = col1.selectbox('Select Season', seasons.keys())
    selected_month = col1.selectbox('Select Month', months.keys())
    selected_hour = col1.selectbox('Select Hour', df['hr'].unique())

    temperature = col2.slider('Temperature (C)', min_value=min_temp, max_value=max_temp, value=default_temp)
    temperature = (temperature - min_temp) / (max_temp - min_temp)

    atemperature = col2.slider('Apparent Temperature (C)', min_value=min_temp, max_value=max_temp, value=default_temp)
    atemperature = (atemperature - min_temp) / (max_temp - min_temp)

    humidity = col2.slider('Humidity (%)', min_value=0, max_value=100, value=50)
    humidity = humidity / 100

    windspeed = col2.slider('Windspeed (km/h)', min_value=0, max_value=max_windspeed, value=default_windspeed)
    windspeed = ((windspeed) / max_windspeed)

    # Add a button to simulate the scenario
    if tab.button('Simulate'):
        with st.spinner('Wait for it...'):
            time.sleep(1)
            # Make prediction using the model
            input_features = [[seasons[selected_season],  # season (e.g., 3 for fall
                            months[selected_month],   # mnth (e.g., 7 for July)
                            selected_hour,  # hr (e.g., 12 for noon)
                            0,   # holiday (e.g., 0 for no holiday)
                            4,   # weekday (e.g., 4 for Thursday)
                            1,   # workingday (e.g., 1 for working day)
                            2,   # weathersit (e.g., 2 for mist + cloudy)
                            temperature,  # temp (normalized temperature)
                            atemperature, # atemp (normalized feeling temperature)
                            humidity, # hum (normalized humidity)
                            windspeed]]  # windspeed (normalized wind speed)

            pred_df = pd.DataFrame(input_features, columns=column_names)
            prediction = model.predict(pred_df)
            tab.write(f'Predicted Bike Counts: {int(prediction[0])}')
    
# Footer
def footer():
    st.sidebar.error('What do we add here??')
    st.sidebar.markdown('---')
    st.sidebar.write('Dashboard created by Group 4')
    st.sidebar.write('Contact: g4@research.ie.edu')

# Recommendations or Actions
# ...

header()

tab1, tab2, tab3, tab4 = st.tabs(['🚴 2012 Results', '📊 Insights', '📈 Will we earn money next week?', 'Simulator'])

results_content(tab1)

insights_content(tab2)

revenue_prediction_content(tab3, column_names)

simulator_content(tab4, model, column_names)

footer()
