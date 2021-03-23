import pandas as pd
import re

jobs = pd.read_csv('allJobs1.csv')

threshold = 0.75

#Dropping columns with missing value rate higher than threshold
jobs = jobs[jobs.columns[jobs.isnull().mean() < threshold]]

#Dropping rows with missing value rate higher than threshold
jobs = jobs.loc[jobs.isnull().mean(axis=1) < threshold]

jobs.loc[jobs['Dropbox Files'].notna(), 'Dropbox Files'] = 1
jobs.loc[jobs['Dropbox Files'].isna(), 'Dropbox Files'] = 0

jobs['bd'] = ''
jobs['bd'] = jobs['bd'].apply(list)
jobs[['Bid Date','bd']] = jobs['Bid Date'].str.split('→ ',expand=True)
jobs = jobs.drop(['bd'], axis = 1)
jobs['Bid Date'] = pd.to_datetime(jobs['Bid Date'])

jobs['pt2'] = ''
jobs['pt2'] = jobs['pt2'].apply(list)
jobs[['Primary Tag','pt2']] = jobs['Primary Tag'].str.split(',', n=1, expand=True)
jobs = jobs.drop(['pt2'], axis = 1)

jobs['t2'] = ''
jobs['t2'] = jobs['t2'].apply(list)
jobs[['Type','t2']] = jobs['Type'].str.split(',', n=1, expand=True)
jobs = jobs.drop(['t2'], axis = 1)

jobs['t2'] = ''
jobs['t2'] = jobs['t2'].apply(list)
jobs[['Locations Bid','t2']] = jobs['Locations Bid'].str.split(',', n=1, expand=True)
jobs = jobs.drop(['t2'], axis = 1)

jobs['t2'] = ''
jobs['t2'] = jobs['t2'].apply(list)
jobs[['Bidder(s)','t2']] = jobs['Bidder(s)'].str.split(',', n=1, expand=True)
jobs = jobs.drop(['t2'], axis = 1)

jobs['t2'] = ''
jobs['t2'] = jobs['t2'].apply(list)
jobs[['Bidding Client','t2']] = jobs['Bidding Client'].str.split(',', n=1, expand=True)
jobs = jobs.drop(['t2'], axis = 1)

jobs['t2'] = ''
jobs['t2'] = jobs['t2'].apply(list)
jobs[['RL_Directors','t2']] = jobs['RL_Directors'].str.split(',', n=1, expand=True)
jobs = jobs.drop(['t2'], axis = 1)

jobs['t2'] = ''
jobs['t2'] = jobs['t2'].apply(list)
jobs[['Agency','t2']] = jobs['Agency'].str.split(',', n=1, expand=True)
jobs = jobs.drop(['t2'], axis = 1)

jobs['t2'] = ''
jobs['t2'] = jobs['t2'].apply(list)
jobs[['Exec Prod / Contact','t2']] = jobs['Exec Prod / Contact'].str.split(',', n=1, expand=True)
jobs = jobs.drop(['t2'], axis = 1)

#Filling missing values with medians of the columns

jobs['Hours'] = jobs['Hours'].fillna(jobs['Hours'].median())
jobs['Bid Cost'] = jobs['Bid Cost'].fillna(jobs['Bid Cost'].median())
jobs['Rate / Avg'] = jobs['Rate / Avg'].fillna(jobs['Rate / Avg'].median())

#filling missing values with max value or "other" for categorical columns

jobs['Bid Date'].fillna(jobs['Bid Date'].value_counts().idxmax(), inplace=True)
jobs['RL_Directors'].fillna(jobs['RL_Directors'].value_counts().idxmax(), inplace=True)
jobs['Insurance'] = jobs['Insurance'].fillna('Other')
jobs['Primary Tag'] = jobs['Primary Tag'].fillna('Other')
jobs['Type'] = jobs['Type'].fillna('Other')
jobs['Locations Bid'] = jobs['Locations Bid'].fillna('Other')
jobs['Status'] = jobs['Status'].fillna('Unknown')
jobs['Agency'] = jobs['Agency'].fillna('Unknown')
jobs['Bidder(s)'] = jobs['Bidder(s)'].fillna('Unknown')
jobs['Exec Prod / Contact'] = jobs['Exec Prod / Contact'].fillna('Other')

# Outlier Detection with Percentiles

upper_lim = jobs['Hours'].quantile(.95)
lower_lim = jobs['Hours'].quantile(.05)
jobs = jobs[(jobs['Hours'] < upper_lim) & (jobs['Hours'] > lower_lim)]

## Binning
# extracting month and year to new columns in df

jobs['month'] = pd.DatetimeIndex(jobs['Bid Date']).month
jobs['year'] = pd.DatetimeIndex(jobs['Bid Date']).year
jobs = jobs.drop(['Bid Date'], axis = 1)

# Numerical Binning

jobs['CostBin'] = pd.cut(jobs['Bid Cost'], bins=[0,663,1310,3925], labels=["Low", "Mid", "High"])

jobs.CostBin = pd.Categorical(pd.factorize(jobs.CostBin)[0] + 1)
jobs['Job Name'] = pd.Categorical(pd.factorize(jobs['Job Name'])[0] + 1)
jobs['Agency'] = pd.Categorical(pd.factorize(jobs['Agency'])[0] + 1)
jobs['Bidder(s)'] = pd.Categorical(pd.factorize(jobs['Bidder(s)'])[0] + 1)
jobs['Bidding Client'] = pd.Categorical(pd.factorize(jobs['Bidding Client'])[0] + 1)
jobs['RL_Directors'] = pd.Categorical(pd.factorize(jobs['RL_Directors'])[0] + 1)
jobs['Insurance'] = pd.Categorical(pd.factorize(jobs['Insurance'])[0] + 1)
jobs['Primary Tag'] = pd.Categorical(pd.factorize(jobs['Primary Tag'])[0] + 1)
jobs['Type'] = pd.Categorical(pd.factorize(jobs['Type'])[0] + 1)
jobs['Locations Bid'] = pd.Categorical(pd.factorize(jobs['Locations Bid'])[0] + 1)
jobs['Exec Prod / Contact'] = pd.Categorical(pd.factorize(jobs['Exec Prod / Contact'])[0] + 1)


def Score(column):
    Won = 'Awarded'
    Done = 'Done'
    #Bidding = 'Bidding'

    for i in column:
        if type(i) == str:
            if re.search(Won, i):
                column.replace(i, 1, inplace=True)
            elif re.search(Done, i):
                column.replace(i, 1, inplace=True)
            #elif re.search(Bidding, i):
                #column.replace(i, 1, inplace=True)
            else:
                column.replace(i, 0, inplace=True)

# categorizing our labels (what we are trying to predict in modeling)
Score(jobs['Status'])

jobs = jobs[['Bidding Client', 'Exec Prod / Contact','RL_Directors','Insurance', 'Agency', 'Bidder(s)',
             'Locations Bid','Dropbox Files','Primary Tag','month','Type','CostBin','Status']]
'''Note: all columns
'Bidding Client', 'Exec Prod / Contact','RL_Directors','Insurance', 'Bidder(s)',
'Agency','Locations Bid','Dropbox Files','Primary Tag','month','Type','Hours','Bid Cost','CostBin','Status'
'''

jobs.to_csv('jobs.csv')
