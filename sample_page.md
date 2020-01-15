## Adult Dataset & their Income Analysis 

**Project description:** 

The "Adult Dataset has around 32,000 records with various information like age, education, marital-status, occupation, gender, hours per week, country and income information". From this dataset I have derived various insights and explained them below.


### 1. Dataset Overview

```
adt=pd.read_csv('adult.csv')
adt.head()

    age         workclass  fnlwgt  ... hours-per-week  native-country income
0   39         State-gov   77516  ...             40   United-States  <=50K
1   50  Self-emp-not-inc   83311  ...             13   United-States  <=50K
2   38           Private  215646  ...             40   United-States  <=50K
3   53           Private  234721  ...             40   United-States  <=50K
4   28           Private  338409  ...             40            Cuba  <=50K
```

### 2. Checking for NULL values in the data
```
adt.isnull().sum()

age               0
workclass         0
fnlwgt            0
education         0
education-num     0
marital-status    0
occupation        0
relationship      0
race              0
sex               0
capital-gain      0
capital-loss      0
hours-per-week    0
native-country    0
income            0
```
No Null values in this dataset.


### 3. Changing the column Names

Existing Column Values: 
Index(['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'],dtype='object')
```
adt.rename(columns={'marital-status' : 'marital'}, inplace = True)
adt.rename(columns={'capital-gain' : 'capgain'}, inplace = True)
adt.rename(columns={'capital-loss' : 'caploss'}, inplace = True)
adt.rename(columns={'hours-per-week' : 'hrsprwk'}, inplace = True)
adt.rename(columns={'native-country' : 'country'}, inplace = True)
```
Corrected Column Values:
Index(['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capgain', 'caploss', 'hrsprwk', 'country', 'income'], dtype='object')


### 4. Grouping the "Countries" into Region

There were 42 unique entries in the country column and around 583 entries where unknow.

```
adt['country'].value_counts()

United-States                 29170
Mexico                          643
?                               583
Philippines                     198
Germany                         137
Canada                          121
Puerto-Rico                     114
El-Salvador                     106
India                           100
Cuba                             95
England                          90
Jamaica                          81
South                            80
China                            75
Italy                            73
Dominican-Republic               70
Vietnam                          67
Guatemala                        64
Japan                            62
Poland                           60
Columbia                         59
Taiwan                           51
Haiti                            44
Iran                             43
Portugal                         37
Nicaragua                        34
Peru                             31
Greece                           29
France                           29
Ecuador                          28
Ireland                          24
Hong                             20
Trinadad&Tobago                  19
Cambodia                         19
Thailand                         18
Laos                             18
Yugoslavia                       16
Outlying-US(Guam-USVI-etc)       14
Hungary                          13
Honduras                         13
Scotland                         12
Holand-Netherlands                1
Name: country, dtype: int64

```
Grouping all the countries into 4 regions (America, Asia, Europe, Others).

Creating a new column 
```
adt['region']='Asia'
```
Grouping the countries
```
adt['region'][adt['country'].isin(['United-States','Canada','Mexico','Puerto-Rico','El-Salvador','Jamaica','Guatemala','Columbia',
    'Nicaragua','Trinadad&Tobago','Cambodia','Laos','Outlying-US(Guam-USVI-etc)','Honduras'])]='America'

adt['region'][adt['country'].isin(['Germany','England','Italy','Dominican-Republic','Poland','Iran','Portugal','France','Greece',
    'Ecuador','Ireland','Scotland','Holand-Netherlands'])]='Europe'

adt['region'][adt['country'].isin(['?',])]='Others'

```
Region Overview
```
adt['region'].value_counts()

America    30475
Asia         870
Europe       633
Others       583
Name: region, dtype: int64
```

### 5. Grouping the "Education Entries" into Categories

There were 16 unique entries in the education column.
```
adt['education'].value_counts()

HS-grad         10501
Some-college     7291
Bachelors        5355
Masters          1723
Assoc-voc        1382
11th             1175
Assoc-acdm       1067
10th              933
7th-8th           646
Prof-school       576
9th               514
12th              433
Doctorate         413
5th-6th           333
1st-4th           168
Preschool          51
Name: education, dtype: int64
```
Grouping all the education into 3 categories (School, Graduate, Other Education)
```
adt['edu_category']='School'
adt['edu_category'][adt['education'].isin(['HS-grad','Some-college','Bachelors','Masters','Doctorate'])]='Graduate'
adt['edu_category'][adt['education'].isin(['Assoc-voc','Assoc-acdm'])]='OthrEDU'
```

Education Category Overview
```
adt['edu_category'].value_counts() 

Graduate    25283
School       4829
OthrEDU      2449
Name: edu, dtype: int64
```

### 6. Grouping the "Occupation" 

There were 15 unique entries in the occupation column.
```
adt['occupation'].value_counts()
 
Prof-specialty       4140
Craft-repair         4099
Exec-managerial      4066
Adm-clerical         3770
Sales                3650
Other-service        3295
Machine-op-inspct    2002
?                    1843
Transport-moving     1597
Handlers-cleaners    1370
Farming-fishing       994
Tech-support          928
Protective-serv       649
Priv-house-serv       149
Armed-Forces            9
Name: occupation, dtype: int64
```
Grouping all the occupation into 2 categories (officeJobs, fieldJobs)
```
adt['jobCategory']='officeJobs'
adt['jobCategory'][adt['occupation'].isin(['Craft-repair','Sales','Transport-moving','Handlers-cleaners','Farming-fishing'])]='fieldJobs'
```
Job Category Overview
```
adt['jobCategory'].value_counts()
 
officeJobs    20851
fieldJobs     11710
Name: occu, dtype: int64
```

### 7. Processing "Hours-per-week" Column

Checking for *Outliers* in the "Hours-per-week" Column.
```
plt.boxplot(adt['hrsprwk'])
```
<img src="images/hrsprwk_outliers.png?raw=true"/>

Treating the *Outliers*

```
p25=adt['hrsprwk'].quantile(.25)
p75=adt['hrsprwk'].quantile(.75)

iqr=p75-p25

lowerrange=p25-(1.5*iqr)
upperrange=p75+(1.5*iqr)

adt['hrsprwk'][adt['hrsprwk']<lowerrange]=np.nan
adt['hrsprwk'][adt['hrsprwk']>upperrange]=np.nan

adt['hrsprwk'].isnull().sum()
adt['hrsprwk'].fillna(0,inplace=True)
```
After treating the *Outliers*.
```
plt.boxplot(adt['hrsprwk'])
```
<img src="images/hrsprwk_outliers_treated.png?raw=true"/>

### 8. Getting Dummies from the newly created column for "Logistic Regression ML Algorithm"

Getting dummies for "Gender" column.
```
s=pd.get_dummies(adt['sex'])
adt=pd.concat([adt,s], axis=1)
```
Getting dummies for "Education Category" column.
```
e=pd.get_dummies(adt['edu_category'])
adt=pd.concat([adt,e],axis=1)
```
Getting dummies for "Gender Category" column.
```
o=pd.get_dummies(adt['jobCategory'])
adt=pd.concat([adt,o],axis=1)
```

Columns list #before# getting dummies
```
adt.columns

Index(['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capgain', 'caploss', 'hrsprwk', 'country', 'income'], dtype='object')
```

Columns list #after# getting dummies
```
adt.columns

Index(['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capgain', 'caploss', 'hrsprwk', 'country', 'income', 'region', 'Female', 'Male', 'edu_category', 'jobCategory', 'Graduate', 'OthrEDU', 'school', 'fieldJobs', 'officeJobs'], dtype='object')



