#Jumana Rahman
import numpy as np
import pandas as pd
import plotly.express as px
from turtle import hideturtle
from scipy.stats import norm
from statistics import mode
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind, ttest_rel, ttest_ind_from_stats

from scipy.stats import norm
from statistics import mode
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
from turtle import hideturtle
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind, ttest_rel, ttest_ind_from_stats
from scipy.stats import norm
from statistics import mode
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from statistics import mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor

import calendar

#################################################################################################################
#load the .csv data
hotelDataOrig = pd.read_csv(r"C:/Users/19172/Desktop/hotel_bookings.csv")
#print(hotelDataOrig)
print("------------------------------------------------")
#drop columns that are definitely not needed for this study
hotelBookingsData = hotelDataOrig.drop(columns = ["is_canceled", "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                                            "babies", "meal", "market_segment",
                                            "distribution_channel", "is_repeated_guest", "previous_cancellations",
                                            "previous_bookings_not_canceled", "reserved_room_type", "assigned_room_type", "booking_changes", "deposit_type", "agent", "company", 
                                            "days_in_waiting_list", "customer_type", "total_of_special_requests", "reservation_status", "reservation_status_date"], axis = 1)
print(hotelBookingsData)
print("------------------------------------------------")

#check for null values
hotelBookingsData = hotelBookingsData.dropna()
print(hotelBookingsData.isnull().any())
print("------------------------------------------------")



#make dataframe for finding patterns in resort hotels and city hotels
resortHotelBookings = hotelBookingsData.query('hotel == "Resort Hotel"')
print(resortHotelBookings)
print("------------------------------------------------")

cityHotelBookings = hotelBookingsData.query('hotel == "City Hotel"')
print(cityHotelBookings)
print("------------------------------------------------")

########################################################################################################################################
# Observe any patterns through visualizations

#statistical data pattern for adr (histogram)
print('statistical data on the adr for city hotels')
print(cityHotelBookings['adr'].describe())
print("------------------------------------------------")
print('statistical data on the adr for resort hotels')
print(resortHotelBookings['adr'].describe())
print("------------------------------------------------")


fig = px.histogram(resortHotelBookings, x="adr") 
fig.show() 

fig2 = px.histogram(cityHotelBookings, x="adr")  
fig2.show() 

#piechart showing amount city hotels vs. resort hotel bookings
plt.rcParams['figure.figsize']=[5,5]
plt.pie(hotelBookingsData.hotel.value_counts().values,labels=hotelBookingsData.hotel.value_counts().index,autopct='%1.1f%%')
plt.title('Types of Hotels')
plt.show()

pd.crosstab(hotelBookingsData.country,hotelBookingsData.hotel).plot.bar()
plt.title("Distribution of Hotel Bookings in Countries")
plt.xlabel('Countries') 
plt.ylabel('Count')
plt.tick_params(labelsize=5)
plt.show()


pd.crosstab(hotelBookingsData.arrival_date_year,hotelBookingsData.hotel).plot.bar()
plt.tick_params(labelsize=5)
plt.title("Distribution of hotel bookings from 2015-2017")
plt.xlabel('Year') 
plt.ylabel('Count')
plt.show()


#bar graph to show whether it city or resort hotels are popular for kids throughout countries
kidshotel = hotelBookingsData.query('children>0.0')
#print(kidshotel)
pd.crosstab(kidshotel.country,kidshotel.hotel).plot.bar()
plt.tick_params(labelsize=5)
plt.title("Hotel Bookings with Kids")
plt.xlabel('Countries') 
plt.ylabel('Count')
plt.show()

#bar graph to show whether city or resort hotels are popular for bookings with only adults throughout countries
onlyadultshotel = hotelBookingsData.query('children==0.0')
#print(onlyadultshotel)
pd.crosstab(onlyadultshotel.country,onlyadultshotel.hotel).plot.bar()
plt.title("Hotel Bookings with Adults Only")
plt.xlabel('Countries') 
plt.ylabel('Count')
plt.tick_params(labelsize=5)
plt.show()

#bar graphs to which months bookings with children were popular for 2015 2016 2017   #done all before ###
kidshotel2015 = kidshotel.query('arrival_date_year==2015')
pd.crosstab(kidshotel2015.arrival_date_month,kidshotel2015.hotel).plot.bar()
plt.tick_params(labelsize=5)
plt.title("Bookings with kids for 2015")
plt.xlabel('Months') 
plt.ylabel('Count')
plt.show()

kidshotel2016 = kidshotel.query('arrival_date_year==2016')
pd.crosstab(kidshotel2016.arrival_date_month,kidshotel2016.hotel).plot.bar()
plt.tick_params(labelsize=5)
plt.title(" Bookings with kids for 2016")
plt.xlabel('Months') 
plt.ylabel('Count')
plt.show()

kidshotel2017 = kidshotel.query('arrival_date_year==2017')
pd.crosstab(kidshotel2017.arrival_date_month,kidshotel2017.hotel).plot.bar()
plt.tick_params(labelsize=5)
plt.title("Bookings with kids for 2017")
plt.xlabel('Months') 
plt.ylabel('Count')
plt.show()


pd.crosstab(kidshotel.arrival_date_month,kidshotel.hotel).plot.bar()
plt.tick_params(labelsize=5)
plt.title("Bookings with kids for Years 2015-2017")
plt.xlabel('Months') 
plt.ylabel('Count')
plt.show()

#bar graph to which months bookings with children were popular throughout all years 2015-2017
pd.crosstab(onlyadultshotel.arrival_date_month,onlyadultshotel.hotel).plot.bar()
plt.tick_params(labelsize=5)
plt.title("Bookings with Adults Only for Years 2015-2017")
plt.xlabel('Months') 
plt.ylabel('Count')
plt.show()
#########################################################################################################################


#make dataframe for training and predicting data (will use hotelBookingsData minus the stays in week and weekends and required parking spaces)
hotelData = hotelBookingsData.drop(columns = ["stays_in_weekend_nights", "stays_in_week_nights", "required_car_parking_spaces"], axis = 1)
#encode columns with names 
le = LabelEncoder()
hotelData['arrival_date_month']= le.fit_transform(hotelData['arrival_date_month']) #July-4 August-1 
hotelData['hotel']= le.fit_transform(hotelData['hotel'])  #resorthotel(1) cityhotel(0)
hotelData['country']= le.fit_transform(hotelData['country']) 
hotelData['arrival_date_year'] = hotelData[['arrival_date_year']].astype(int)
print(hotelData)
print("------------------------------------------------")


#print(hotelBookingsData.corr())
#####################################################################################################################################3
#Gaussian Model to predict what the next person will book using Gaussian Naive Bayes
print("Gaussian Model")
x = hotelData.iloc[:,[1,2,3,4,5,6]].values  #independent variable
#print(x)
print("------------------------------------------------")

y = hotelData.iloc[:,0].values      #target variable
#print(y)
print("------------------------------------------------")

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
print(x_train[:5]) #print first five rows of trained data

print("------------------------------------------------")
mean_arrival_date_year = x_train[:,0].mean()
std_arrival_date_year = x_train[:,0].std()

mean_arrival_date_month = x_train[:,1].mean()
std_arrival_date_month = x_train[:,1].std()

mean_adults = x_train[:,2].mean()
std_adults = x_train[:,2].std()

mean_children = x_train[:,3].mean()
std_children = x_train[:,3].std()

mean_country = x_train[:,4].mean()
std_country = x_train[:,4].std()

mean_adr = x_train[:,5].mean()
std_adr = x_train[:,5].std()
#_______________________________________

#Standardization
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
#print(x_train[0,0])   # we get same result as line 52: -1.0

x_test = sc.transform(x_test) #to perform centering and scaling for xtest

#Now to import our Gaussian Model
gnb = GaussianNB()
gnb.fit(x_train,y_train) #train the model using the training set

y_predict = gnb.predict(x_test)
print(y_predict)

#____________________________________
#evaluatet the model's output
print("------------------------------------------------")
#calculate accuracy
print("The accuracy of the Gaussian Model is: ",accuracy_score(y_test, y_predict))
print("------------------------------------------------")

#Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix: \n")
print(cm)
print("------------------------------------------------")

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity: ", sensitivity)
print("------------------------------------------------")

specificity = cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity: ", specificity)
print("------------------------------------------------")

#Prediction
print(gnb.predict([[-1,0.8,0.8,0.8,0.8, 0.5]]))

arrival_date_year_p = (-1)*std_arrival_date_year +mean_arrival_date_year
print("Year of Arrival: ", round(arrival_date_year_p))

arrival_date_month_p = (-1)*std_arrival_date_month +mean_arrival_date_month
print("Month of Arrival: ", round(arrival_date_month_p))

adults_p = (-1)*std_adults +mean_adults
print("Number of Adults: ", round(adults_p))

children_p = (-1)*std_children +mean_children
print("Number of Children: ", round(children_p))

country_p = (-1)*std_country +mean_country
print("Country: ", round(country_p))

adr_p = (-1)*std_adr +mean_adr
print("Average Daily Rate: ", round(adr_p))
print("------------------------------------------------")
print("------------------------------------------------")

#ex = hotelData.query('country == 48')
#print(ex)
###############################################################################################################
#OLS Regression
print("Regression Model 1 with three variables")
print("correlation coefficients below:")
print(hotelData.corr())   #strongest top four variables with hotel is children, country, adr

independent = ['children', 'country', 'adr']
x = hotelData[independent]

#check VIF
print("VIF DATA BELOW")
vif_data= pd.DataFrame()
vif_data["features"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)
print("------------------------------------------------")

#check OLS
y = hotelData['hotel']

print('Regression with all three variables:')
sm_var = sm.add_constant(x)
mlr_model = sm.OLS(y, sm_var)
mlr_reg = mlr_model.fit()
print(mlr_reg.summary())

#all variables Linear Regression: Sex, Age, and Fare
     #train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state= 0)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("------------------------------------------------")
print(model.score(x_test, y_test))
print("------------------------------------------------")

 # Now make the prediction using the model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)
y_pred = model.predict(x_test)
y_pred = y_pred.round()
df = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
print(df)
print("------------------------------------------------")


print("Make prediction:")
y_pred = model.predict(x_test)
y_pred = y_pred.round()
print(y_pred) 


###############################################################################################################

print("Regression Model 2 with two variables (children and adr)")



print('Regression with two variables(children and adr):')

independent = ['children', 'adr']

x_a = hotelData[independent]
sm_var = sm.add_constant(x_a)
mlr_model = sm.OLS(y, sm_var)
mlr_reg = mlr_model.fit()
print(mlr_reg.summary())
    #train and test
x_train, x_test, y_train, y_test = train_test_split(x_a, y, train_size=0.7, random_state= 0)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("------------------------------------------------")
print(model.score(x_test, y_test))
print("------------------------------------------------")

 # Now make the prediction using the model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)
y_pred = model.predict(x_test)
y_pred = y_pred.round()
df = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
print(df)
print("------------------------------------------------")


print("Make prediction:")
y_pred = model.predict(x_test)
y_pred = y_pred.round()
print(y_pred) 
