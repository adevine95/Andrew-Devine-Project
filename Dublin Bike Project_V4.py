#%%
import pandas as pd
import folium
import datetime 
import numpy as np
from sympy.plotting.tests.test_plot import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
#%%
data_2019_q1 = pd.read_csv('dublinbikes_20190101_20190401.csv')
data_2019_q2 = pd.read_csv('dublinbikes_20190101_20190401.csv')
data_2019_q3 = pd.read_csv('dublinbikes_20190101_20190401.csv')
data_2019_q4 = pd.read_csv('dublinbikes_20190101_20190401.csv')
data_2019 = pd.concat([data_2019_q1,data_2019_q2,data_2019_q3,data_2019_q4],axis=0)

data_2020_q1 = pd.read_csv('dublinbikes_20200101_20200401.csv')
data_2020_q2 = pd.read_csv('dublinbikes_20200101_20200401.csv')
data_2020_q3 = pd.read_csv('dublinbikes_20200101_20200401.csv')
data_2020_q4 = pd.read_csv('dublinbikes_20200101_20200401.csv')
data_2020 = pd.concat([data_2020_q1,data_2020_q2,data_2020_q3,data_2020_q4],axis=0)

data_2021_q1 = pd.read_csv('dublinbikes_20210101_20210401.csv')
data_2021_q2 = pd.read_csv('dublinbikes_20210101_20210401.csv')
data_2021_q3 = pd.read_csv('dublinbikes_20210101_20210401.csv')
data_2021_q4 = pd.read_csv('dublinbikes_20210101_20210401.csv')
data_2021 = pd.concat([data_2021_q1,data_2021_q2,data_2021_q3,data_2021_q4],axis=0)



#%%
def clean_data(data_to_run):
    '''
    Clean column names and split out times
    '''
    data_to_run=data_to_run[data_to_run['STATUS']== 'Open']
    data_to_run=data_to_run[data_to_run['STATION ID']!=507]
    data_to_run.rename(columns=
            {'STATION ID':'station_id',
                'TIME':'timestamp', 
                'NAME':'name',
                'BIKE STANDS':'no_stands', 
                'AVAILABLE BIKE STANDS':'free_stands', 
                'AVAILABLE BIKES':'free_bikes',
                'LAST UPDATED':'last_updated', 
                'STATUS':'status',
                'ADDRESS':'address',
                'LATITUDE':'latitude',
                'LONGITUDE':'longitude'
                },inplace=True)

    data_to_run.drop_duplicates(keep= 'first',inplace=True)
    data_to_run['proportion_filled'] = (data_to_run['free_bikes']/data_to_run['no_stands'])

    #Change times from strings to datetimes
    data_to_run['timestamp']= pd.to_datetime((data_to_run['timestamp']),format ='%Y-%m-%d %H:%M:%S.%f' )
    #split out individual time attributes 

    data_to_run["date"] = [d.date() for d in data_to_run["timestamp"]]
    data_to_run["time"] = [d.time() for d in data_to_run["timestamp"]]
    data_to_run['year'] = pd.DatetimeIndex(data_to_run['timestamp']).year
    # data_to_run['month'] = pd.DatetimeIndex(data_to_run['timestamp']).month
    data_to_run['hour'] = pd.DatetimeIndex(data_to_run['timestamp']).hour
    data_to_run['time_of_day'] = pd.DatetimeIndex(data_to_run['timestamp']).hour
    # data_to_run['minute'] = pd.DatetimeIndex(data_to_run['timestamp']).minute
    data_to_run['day_num'] = pd.DatetimeIndex(data_to_run['timestamp']).weekday
    data_to_run['day_name'] = pd.DatetimeIndex(data_to_run['timestamp']).weekday
    data_to_run['workday'] = pd.DatetimeIndex(data_to_run['timestamp']).weekday
    data_to_run.drop(['timestamp'], axis = 1)

    #rename some data using dictionaries
    day_str = ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
    day_num = (0,1,2,3,4,5,6)
    day_zip = zip(day_num,day_str)
    day_list= list(day_zip)
    day_dict = dict(day_list)

    workday_str = ('midweek','midweek','midweek','midweek','midweek','weekend','weekend')
    workday_zip = zip(day_num,workday_str)
    workday_list= list(workday_zip)
    workday_dict = dict(workday_list)

    hour_str = ('night','night','night','night','night','night','morning','morning','morning','morning','morning','midday','midday','midday','midday','midday','evening','evening','evening','evening','night','night','night','night')
    hour_num = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
    hour_zip = zip(hour_num,hour_str)
    hour_list= list(hour_zip)
    hour_dict = dict(hour_list)

    data_to_run['day_name'].replace(day_dict,inplace=True)
    data_to_run['workday'].replace(workday_dict,inplace=True)
    data_to_run['workday'] = data_to_run['workday'].astype('category')
    data_to_run['time_of_day'].replace(hour_dict,inplace=True)
    data_to_run['time_of_day'] = data_to_run['time_of_day'].astype('category')

    '''Identify days when they recirculate the bikes 
    '''
    # identify bikes coming and going
    data_to_run['bike_change'] = data_to_run.groupby('station_id')['free_stands'].diff(-1)
    data_to_run['bikes_put_in'] = np.where(data_to_run['bike_change'] > 0, data_to_run['bike_change'], 0)
    data_to_run['bikes_taken_out'] = np.where(data_to_run['bike_change'] < 0, data_to_run['bike_change'], 0)
    data_to_run['activity'] = np.where(abs(data_to_run['bike_change']) >= 10, "recirculate", "personal_use")
    data_to_run['too_full/empty'] = np.where(data_to_run['proportion_filled'] < .1, 1, np.where(data_to_run['proportion_filled'] > .9, 1,0 ))

    # Identify timestamps with recirculating and drop them from the dataframe
    data_to_run['recirculating'] = np.where(data_to_run['activity'] == 'recirculate', 1,0)
    data_to_run['join_on'] = data_to_run['station_id'].apply(str)  + (data_to_run['date']).apply(str) 
    join_table= data_to_run.groupby(['join_on'])['recirculating'].sum()
    data_to_run = data_to_run.drop(['recirculating'], axis = 1)
    join_table = join_table.to_frame()
    join_table =join_table.reset_index()
    data_to_run = pd.merge(data_to_run, join_table, on = 'join_on', how = 'left')
    data_to_run = data_to_run.drop(['join_on'], axis = 1)

    return data_to_run

#%%
'''Run function to clean the data'''
data_2019_clean = clean_data(data_2019)
data_2020_clean = clean_data(data_2020)
data_2021_clean = clean_data(data_2021)


#%%
def create_Kmeans_clusters(data_to_run):
    '''Run Kmeans clustering'''

    #group data into clusters
    data_to_run['cluster_group'] =  data_to_run['workday'].astype(str) + data_to_run['time_of_day'].astype(str) 
    df_personal_use= data_to_run.loc[(data_to_run['activity']=='personal_use')]
    df_kmeans = df_personal_use[['station_id', 'name', 'latitude', 'longitude', 'workday', 'time_of_day', 'proportion_filled','cluster_group']]
    df_kmeans = df_kmeans.groupby(['station_id', 'name', 'latitude', 'longitude', 'cluster_group'],as_index=False)['proportion_filled'].mean()
    df_kmeans = df_kmeans.set_index('station_id')
    #pivot dataframe for clustering
    df_kmeans = df_kmeans.pivot_table(index= ['name', 'station_id','latitude', 'longitude'] , columns=['cluster_group'], values='proportion_filled')
    df_kmeans  = df_kmeans.reset_index()
    df_kmeans  = df_kmeans .set_index('name')
    df_kmeans = df_kmeans.dropna()
    return df_kmeans
#%%
'''Manipulate data for Kmeans clustering'''
Kmeans_2019 = create_Kmeans_clusters(data_2019_clean)
Kmeans_2020 = create_Kmeans_clusters(data_2020_clean)
Kmeans_2021 = create_Kmeans_clusters(data_2021_clean)


#%%
def elbow_test(df_kmeans):
    '''Elbow Method - finding the optimal K '''
    distortions = []
    K = range(2,10)
    X = np.array(df_kmeans.drop(['station_id', 'latitude', 'longitude'], 1).astype(float))
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

        
    plt.figure(figsize=(10,7))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return plt.show()
#%%
'''Run the elbow test for Kmeans clustering'''
elbow_test_2019 = elbow_test(Kmeans_2019)
elbow_test_2020 = elbow_test(Kmeans_2020)
elbow_test_2021 = elbow_test(Kmeans_2021)

#%%
#clustering algo
'''Function to run Kmeans clustering'''
def plot_clusters(df_clusters):
    X = np.array(df_clusters.drop(['station_id', 'latitude', 'longitude'], 1).astype(float))
    KM = KMeans(n_clusters=5) 
    KM.fit(X)
    clusters = KM.predict(X)

    locations = df_clusters
    locations['cluster'] = clusters
    locations = locations.reset_index()

    return locations
#%%
'''Run Kmeans Clustering'''
plot_clusters_2019 = plot_clusters(Kmeans_2019)
plot_clusters_2020 = plot_clusters(Kmeans_2020)
plot_clusters_2021 = plot_clusters(Kmeans_2021)

#%%
'''Plot the clusters on a map of Dublin'''
plots = [plot_clusters_2019,plot_clusters_2020,plot_clusters_2021]
for i in plots:
    colordict = {0: 'blue', 1: 'red', 2: 'orange', 3: 'green', 4: 'purple'}
    bstreet = (53.35677,-6.26814)
    dublin_map = folium.Map(location = bstreet,
                            zoom_start=12)
    for LATITUDE, LONGITUDE, cluster in zip(i['latitude'],i['longitude'], i['cluster']):
        folium.CircleMarker(
            [LATITUDE, LONGITUDE],
            color = 'b',
            radius = 8,
            fill_color=colordict[cluster],
            fill=True,
            fill_opacity=0.9
            ).add_to(dublin_map)
    display(dublin_map)
    
#%%
df_cluster_ids=df_kmeans[['station_id','cluster']].drop_duplicates()
df_cluster_ids = df_cluster_ids.sort_values('station_id')
df_cluster_ids=df_cluster_ids.reset_index()
df_cluster_ids = df_cluster_ids[['station_id','cluster']]
data_to_run_clusters = data_to_run.merge(df_cluster_ids,on='station_id',how='left')


    

# %%
#%% 
'''boxplot 9am Vs 5pm'''
# df_midday = data_to_run.loc[
#    #(data_to_run['month']==12)&
#    ((data_to_run['hour']==9)|(data_to_run['hour']==18))&
#    (data_to_run['minute']==0)&
#    ((data_to_run['id']==69))
#    ]

# sns.set_style('darkgrid')
# sns.boxplot(x='hour', y='proportion_filled',data=df_midday,
#            hue='name'
#            )
# plt.ylim(0,100)
#%%
'''boxplot time of day by midweek Vs weekend '''
# df_time_of_day = data_to_run.loc[
#    ((data_to_run['id']==69))
#    ]
# sns.set_style('darkgrid')
# sns.boxplot(x='time_of_day', y='proportion_filled',
#             data=df_time_of_day,
#             order=['morning','midday','afternoon','evening','night'],
#            hue='workday'
#            )
# plt.ylim(0,100)
#%% 
'''
relplot showing % of bikes in a each stand over the course of a day
I'm hoping that U shaped graphs will show origin stations that people leave during the day ie. peoples homes
n shaped graphs should show destination stations that people arrive at in the morning and leave from at night 
plotting the workday as the hue we should see changes in the stations only used for commuting
'''
# sns.set_style('darkgrid')
# sns.relplot(
#    x='hour',
#    y='proportion_filled',
#    kind = 'line',
#    data=data_to_run,
#    col='id',
#    col_wrap=10,
   
#            )

# plt.ylim(0,100)
# sns.set_style('darkgrid')
# sns.relplot(
#    x='hour',
#    y='proportion_filled',
#    kind = 'line',
#    data=data_to_run,
#    col='id',
#    col_wrap=10,
#    hue = 'workday'
#            )
# plt.ylim(0,100)

# palette = {0: 'blue', 1: 'red', 2: 'orange', 3: 'green',4:'purple'}
# sns.set_style('darkgrid')
# sns.relplot(
# x='hour',
# y='proportion_filled',
# kind = 'line',
# data=data_to_run_clusters,
# col='station_id',
# col_wrap=10,
# hue = 'cluster',
# palette = palette
#         )
# plt.ylim(0,1)
# %%

'''Bring in map, centered on Dublin city '''
# bstreet = (53.35677,-6.26814)
# df_coords=data_to_run.loc[:, ['station_id','name', 'latitude', 'longitude']].drop_duplicates().values  
# df_map=pd.DataFrame(df_coords, columns=['station_id','name', 'latitude', 'longitude'])
# map = folium.Map(location=bstreet, zoom_start=12)
# map
# # %%
# '''Map the coordinates of each station and label them'''
# for i in range(0,len(df_map)):
#    folium.Marker(
#       location=[df_map.iloc[i]['latitude'], df_map.iloc[i]['longitude']],
#       popup=df_map.iloc[i]['name'],
#    ).add_to(map)
# '''show map again'''
# map