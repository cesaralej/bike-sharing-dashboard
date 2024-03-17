import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import requests
import time

st.set_page_config(page_title='BiciWash BI Dashboard', page_icon='🚴', layout='wide')

# Local variables
column_names = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
max_temp = 41
max_atemp = 50
default_temp = 15
max_windspeed = 67
default_windspeed = 10
total_bikes = 1000

# Load the dataset
df = pd.read_csv('data/bike-sharing_hourly.csv')

# Load the model
with open('models/reg.pickle', 'rb') as file:
    model = pickle.load(file)

# Visualize the bike counts for a selected month and compare it to the previous year
def get_monthly_graph(dataframe, selected_month):
    dataframe['dteday'] = pd.to_datetime(dataframe['dteday'])
    daily_counts = df.groupby(['dteday'])['cnt'].sum().reset_index()
    monthly_y1_df = daily_counts[daily_counts['dteday'].dt.month == selected_month][daily_counts['dteday'].dt.year == 2011].reset_index(drop=True)
    monthly_y2_df = daily_counts[daily_counts['dteday'].dt.month == selected_month][daily_counts['dteday'].dt.year == 2012].reset_index(drop=True)
    monthly_y2_df['2011'] = monthly_y1_df['cnt']
    monthly_y2_df['2012'] = monthly_y2_df['cnt']
    compare_graph = px.line(monthly_y2_df, x='dteday', y=['2012', '2011'],
                labels={'cnt': 'Total Bike Counts', 'dteday': 'Date', 'year': 'Year'},
                template='ggplot2')
    return compare_graph

# Visualize the bike counts for 2012 and compare it to 2011
def get_yearly_graph(dataframe):
    dataframe['dteday'] = pd.to_datetime(dataframe['dteday'])
    daily_counts = dataframe.groupby(['dteday'])['cnt'].sum().reset_index()
    monthly_y1_df = daily_counts[daily_counts['dteday'].dt.year == 2011].reset_index(drop=True)
    monthly_y2_df = daily_counts[daily_counts['dteday'].dt.year == 2012].reset_index(drop=True)
    monthly_y2_df['2011'] = monthly_y1_df['cnt']
    monthly_y2_df['2012'] = monthly_y2_df['cnt']
    compare_graph = px.line(monthly_y2_df, x='dteday', y=['2012', '2011'],
                labels={'cnt': 'Total Bike Counts', 'dteday': 'Date', 'year': 'Year'},
                template='ggplot2')
    return compare_graph

# Visualize the average temperature and bike counts for 2012
def get_yearly_avg_temp_graph(dataframe):
    dataframe['dteday'] = pd.to_datetime(dataframe['dteday'])
    daily_counts = dataframe.groupby(['dteday'])['cnt'].sum().reset_index()
    avg_temp = dataframe.groupby(['dteday'])['temp'].mean().reset_index()
    monthly_y2_df = daily_counts[daily_counts['dteday'].dt.year == 2012].reset_index(drop=True)
    monthly_y2_df['temp'] = avg_temp['temp'].apply(lambda x: x * daily_counts['cnt'].mean()*2)
    monthly_y2_df['2012'] = monthly_y2_df['cnt']
    compare_graph = px.line(monthly_y2_df, x='dteday', y=['2012', 'temp'],
                            labels={'2012': 'Total Bike Counts', 'temp': 'Temperature', 'dteday': 'Date'},
                            template='ggplot2')
    return compare_graph

# Function to visualize the average bike counts by hour for each day of the week
def plot_average_bike_counts_by_hour(dataframe):
    # Dictionary to map weekday number to name
    weekday_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}

    # Calculate the average count of bikes for each day of the week and hour
    avg_counts = dataframe.groupby(['weekday', 'hr'])['cnt'].mean().reset_index()

    # Replace numeric weekday values with corresponding names
    avg_counts['weekday'] = avg_counts['weekday'].map(weekday_names)

    # Create a Plotly Express line chart
    fig = px.line(avg_counts, x='hr', y='cnt', color='weekday',
                  labels={'cnt': 'Average Bike Counts', 'hr': 'Hour of the Day', 'weekday': 'Day of the Week'})

    return fig

# Function to prepare API data for prediction
def prepare_api_dataframe(response, col_names):
    weather_df = pd.DataFrame(response.json()['hourly'])
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df['mnth'] = weather_df['time'].dt.month
    weather_df['hr'] = weather_df['time'].dt.hour
    weather_df['temp'] = (weather_df['temperature_80m']) / (max_temp)
    weather_df['atemp'] = (weather_df['apparent_temperature']) / (max_atemp)
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

# Fetch weather information from an API
def weather_request(col_names):
    url = "https://api.open-meteo.com/v1/forecast?latitude=38&longitude=-77&hourly=relative_humidity_2m,apparent_temperature,rain,snowfall,wind_speed_10m,temperature_80m&timezone=America%2FNew_York"

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
    col1.image('images/Artboard – 3.png', use_column_width=True)
    col2.title('BiciDC BI Dashboard')
    col2.write("Welcome to BiciWash's analysis and prediction dashboard.")
    col2.write("BiciDC is the eBike provisioning company that operates for Washington D.C.'s public transport department. Data from 2011 and 2012 has been used for this dashboard.")

# Function to get the month with the most rentals
def get_month_max_rentals(df, year):
    monthly_rentals = df[df['yr'] == year].groupby(['mnth'])['cnt'].sum()
    max_monthly_rentals = monthly_rentals.idxmax()
    max_monthly_rentals_count = monthly_rentals.max()
    max_monthly_rentals_count = "{:,.0f}".format(max_monthly_rentals_count)

    dict_month = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    return dict_month[max_monthly_rentals]

# Function to get the weekday with the most average rentals
def get_weekday_max_rentals(df, year):
    # Filter data for the specified year
    year_data = df[df['yr'] == year]
    
    # Group by weekday and calculate average rentals
    avg_rentals_by_weekday = year_data.groupby('weekday')['cnt'].mean()
    
    # Find the weekday with the maximum average rentals
    max_avg_rentals_weekday = avg_rentals_by_weekday.idxmax()
    max_avg_rentals_count = avg_rentals_by_weekday.max()
    max_avg_rentals_count = "{:,.0f}".format(max_avg_rentals_count)
    
    # Dictionary to map weekday number to name
    dict_weekday = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 0: 'Sunday'}
    
    # Generate result string
    result = f"The weekday with the most average rentals was {dict_weekday[max_avg_rentals_weekday]} with {max_avg_rentals_count} rentals on average"
    
    return dict_weekday[max_avg_rentals_weekday]

# Function to get the hour with the most average rentals for weekdays or weekends
def get_hour_max_rentals(df, year, is_weekday=True):
    # Filter data for the specified year and the selected days (weekdays or weekends)
    if is_weekday:
        selected_days = [ 1, 2, 3, 4, 5]  # Weekdays: Monday to Friday
    else:
        selected_days = [0, 6]  # Weekends: Saturday and Sunday
    
    year_selected_days_data = df[(df['yr'] == year) & df['weekday'].isin(selected_days)]
    
    # Group by hour and calculate average rentals
    avg_rentals_by_hour = year_selected_days_data.groupby('hr')['cnt'].mean()
    
    # Find the hour with the maximum average rentals
    max_avg_rentals_hour = avg_rentals_by_hour.idxmax()
    max_avg_rentals_count = avg_rentals_by_hour.max()
    max_avg_rentals_count = "{:,.0f}".format(max_avg_rentals_count)  # Format count
    
    return f"{max_avg_rentals_hour}:00"

# Function to display the results content
def results_content(tab):
    tab.header('Yearly Overview')
    tab.write("Let's look at our results from 2012.")

    col1, col2, col3, col4, col5 = tab.columns(5)

    selected_year = col1.selectbox('Select Year', [2011, 2012], index=1)
    year_value = 1 if selected_year == 2012 else 0

    # Month with most rentals
    col2.metric(label="Best Month", value=get_month_max_rentals(df, year_value))

    # Week day with most average rentals
    col3.metric(label="Best Day on Average", value=get_weekday_max_rentals(df, year_value))

    # Weekday hour with most average rentals
    col4.metric(label="Peak Weekday Hour", value=get_hour_max_rentals(df, year_value, True))

    # Weekend day with most average rentals
    col5.metric(label="Peak Weekend Hour", value=get_hour_max_rentals(df, year_value, False))
    
    tab.subheader('Bike Counts by Year')
    tab.write('The graph below shows the bike counts for 2012 and compares them to 2011. You can also select a specific month to see the bike counts for that month.')
    
    # Choose a month
    months = {'All Months':0, 'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    selected_month = tab.selectbox('Select Month', months.keys(), index=0)
    month = months[selected_month]

    if month == 0:
        yearly_graph = get_yearly_graph(df)
        tab.plotly_chart(yearly_graph)
    else:
        monthly_graph = get_monthly_graph(df, month)
        tab.plotly_chart(monthly_graph)

# Function to display the insights content
def insights_content(tab):
    tab.header('Descriptive Analytics')
    tab.write("Let's take a look at some insights from the bike-sharing data for 2012.")

    tab.subheader('Average Temperature vs Bike Counts')
    tab.write('The graph below shows the average temperature and bike counts for 2012. The temperature Y-axis was scaled up to match the bike counts. We can see that there is a correlation between temperature and bike counts.')
    tab.plotly_chart(get_yearly_avg_temp_graph(df))

    tab.subheader('Average Bike Counts by Hour')
    tab.write('The graph below shows the average bike counts by hour for each day of the week. We can see that the bike counts peak during the morning and evening rush hours in week days.')
    tab.plotly_chart(plot_average_bike_counts_by_hour(df))

# Function to display the weekly revenue simulator content
def revenue_prediction_content(tab, col_names, dataframe, total_bikes):
    # Predictive Analytics Section
    tab.header('Predictive Analytics')
    tab.write("Grabbing the Washington D.C. weather forecast from https://open-meteo.com open-source API, we are able to predict the total number of bicycle users on an hourly basis for the next 7 days and our earnings using a machine learning model.")
    tab.write("Depending on the predicted count of bikes and our earnings, the app will suggest business strategy recommendations.")
    tab.subheader('Select Revenue and Operational Cost')
    col1, col2 = tab.columns([2,5])
    Revenue = col1.number_input('Revenue per Bike per Hour ($)', min_value=0.00, max_value=10.00, value=5.00)
    OperationalCost = col1.number_input('Operational Cost per Hour ($)', min_value=300.00, max_value=1000.00, value=450.00)

    # Send request to the weather API and make predictions
    if tab.button('Press to Predict Earnings'):
        with st.spinner('Wait for it...'):
            time.sleep(2)

            forecast = weather_request(col_names)
            filtered_forecast = forecast.drop(columns=['time'])
            prediction_w1 = model.predict(filtered_forecast)
            prediction_w1 = prediction_w1.clip(min=0)
            complete_df = pd.concat([forecast, pd.DataFrame(prediction_w1, columns=['predicted_cnt'])], axis=1)
            complete_df['earnings'] = (complete_df['predicted_cnt'] * Revenue) - OperationalCost

            total_earnings = complete_df['earnings'].sum()
            max_count = complete_df['predicted_cnt'].max()
            average_count = complete_df['predicted_cnt'].mean()
            total_earnings_f = "{:,.2f}".format(total_earnings)

            # Get the first day of the forecast
            fc_month = complete_df.head(1)['mnth'].values[0]
            fc_day = complete_df.head(1)['time'].dt.day.values[0]

            # Compare it to the last 7 days of the dataset
            last_7_days = dataframe[dataframe['yr'] == 1]
            last_7_days['dteday'] = pd.to_datetime(last_7_days['dteday'])
            last_7_days['date'] = last_7_days['dteday'] + pd.to_timedelta(last_7_days['hr'], unit='h')
            last_7_days = last_7_days[['date', 'cnt']]
            last_7_days = last_7_days[last_7_days['date'] >= pd.to_datetime(f'2012-{fc_month}-{fc_day}')]
            last_7_days.reset_index(drop=True, inplace=True)
            last_7_days = last_7_days.head(24*7)
            complete_df['previous_cnt'] = last_7_days['cnt']
            complete_df.rename(columns={'predicted_cnt': 'Prediction', 'previous_cnt': '2012 Count'}, inplace=True)
            
            tab.divider()
            tab.subheader('Prediction Results')
            col3, col4, col5 = tab.columns(3)
            col3.metric('Average Bike Counts', int(average_count))
            col4.metric('Max Bike Counts', int(max_count))

            tab.subheader('Recommendations')
            if total_earnings < 0:
                col5.metric('Total Earnings', f'${total_earnings_f}', f'-${total_earnings_f[1:]}')
                tab.warning('We are not expecting to break even in the next 7 days. We should consider user acquisition strategies like investing in a marketing campaign.')
                st.snow()
            else:
                col5.metric('Total Earnings', f'${total_earnings_f}', f'${total_earnings_f}')
                tab.success('We are expecting revenue in the next 7 days. Keep up the good work!')
                st.balloons()
            
            # Compare max count to total bikes
            if max_count > total_bikes*0.8:
                tab.warning('The maximum count is higher than 80% of the total bikes. Consider increasing the price per hour or investing in more bikes.')
            elif max_count < total_bikes*0.5:
                tab.success('The maximum count is lower than 50% of the total bikes. We can consider moving some bikes to another city this week or increase the number of bikes in maintenance.')
            else:
                tab.success('The maximum count is between 80% and 50% of the total bikes. We should be fine this week')

            tab.subheader('7-Day Count Forecast')
            tab.write('The graph below shows the forecasted bike counts for the next 7 days and how it compares to the values from the same week in 2012.')

            # Create a line plot using Plotly Express
            fig = px.line(complete_df, x='time', y=['Prediction', '2012 Count'], labels={'value': 'Count', 'date': 'Date'},
                         template='ggplot2')
            tab.plotly_chart(fig)

# Function to display the simulator content
def simulator_content(tab, model, col_names):
    tab.header('Simulator')
    tab.write("We can use this section to simulate different scenarios and see how much we can earn.")

    tab.header('Prediction Input')
    col1, col2 = tab.columns(2)

    seasons = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}
    months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    weather_situations = {'Clear': 1, 'Mist + Cloudy': 2, 'Light Snow + Light Rain': 3, 'Heavy Rain + Ice Pallets + Thunderstorm + Mist': 4}
    weekdays = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    # Define seasons and months dictionaries
    seasons_filter = {'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Fall': [9, 10, 11], 'Winter': [12, 1, 2]}
    months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    # Create selection boxes for season and month
    selected_season = col1.selectbox('Select Season', list(seasons.keys()))
    selected_month = col1.selectbox('Select Month', [month for month, num in months.items() if num in seasons_filter[selected_season]])
    weekday = col1.selectbox('Weekday', weekdays.keys())
    workingday = 0 if weekday in ['Saturday', 'Sunday'] else 1
    selected_hour = col1.selectbox('Select Hour', df['hr'].unique(), index=12)
    holiday = col1.radio('Holiday', ['No', 'Yes'])
    holiday = 1 if holiday == 'Yes' else 0
    selected_weather = col1.selectbox('Select Weather Situation', weather_situations.keys())

    temperature = col2.slider('Temperature (°C)', min_value=0, max_value=max_temp, value=default_temp)
    temperature = (temperature) / (max_temp)

    atemperature = col2.slider('Apparent Temperature (°C)', min_value=0, max_value=max_atemp, value=default_temp)
    atemperature = (atemperature) / (max_atemp)

    humidity = col2.slider('Humidity (%)', min_value=0, max_value=100, value=50)
    humidity = humidity / 100

    windspeed = col2.slider('Windspeed (km/h)', min_value=0, max_value=max_windspeed, value=default_windspeed)
    windspeed = ((windspeed) / max_windspeed)
    
    # Add a button to simulate the scenario
    if tab.button('Simulate'):
        with st.spinner("Calculating..."):
            time.sleep(1)
            # Make prediction using the model
            input_features = [[seasons[selected_season],  # season (e.g., 3 for fall
                            months[selected_month],   # mnth (e.g., 7 for July)
                            selected_hour,  # hr (e.g., 12 for noon)
                            holiday,   # holiday (e.g., 0 for no holiday)
                            weekdays[weekday],   # weekday (e.g., 4 for Thursday)
                            workingday,   # workingday (e.g., 1 for working day)
                            weather_situations[selected_weather],   # weathersit (e.g., 2 for mist + cloudy)
                            temperature,  # temp (normalized temperature)
                            atemperature, # atemp (normalized feeling temperature)
                            humidity, # hum (normalized humidity)
                            windspeed]]  # windspeed (normalized wind speed)

            pred_df = pd.DataFrame(input_features, columns=col_names)
            prediction = model.predict(pred_df)
            prediction = prediction.clip(min=0)
            tab.metric('Predicted Bike Counts', int(prediction[0]))
    
# Footer
def footer():
    # Define company information
    company_name = "BiciWash - Group 4"
    company_logo = "images/Artboard – 3.png"
    company_address = "123 Main Street, Washington, D.C."
    company_email = "info@biciwash.com"
    company_phone = "+1 (123) 456-7890"

    # Create the footer layout
    st.markdown("---")
    col1, col2 = st.columns([1, 5])
    col2.write(f"© 2024 {company_name}. All rights reserved.")
    col2.write(
        f"Contact: {company_address} | Email: [{company_email}](mailto:{company_email}) | Phone: {company_phone}"
    )
    col2.write(f"Follow us: [Yahoo Humans](https://www.snapchat.com) | [MySpace 2](https://www.instagram.com) | [MSN Messenger Stories](https://www.facebook.com)")
    col1.image(company_logo, width=100)

# Main content
header()
tab1, tab2, tab3, tab4 = st.tabs(['🚴 Yearly Overview', '📊 Insights', '📈 Predictions and Recommendations', '🎮 Simulator'])
results_content(tab1)
insights_content(tab2)
revenue_prediction_content(tab3, column_names, df, total_bikes)
simulator_content(tab4, model, column_names)
footer()
