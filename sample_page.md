## Adult Dataset & their Income Analysis 

**Project description:** 

The "Adult Dataset has around 32,000 records with various information like age, education, marital-status, occupation, gender, hours per week, country and income information". From this dataset I have derived various insights and explained them below.

<button class="tablink" onclick="openPage('Home', this, 'red')" id="defaultOpen">Exploratory Data Analysis</button>
<button class="tablink" onclick="openPage('News', this, 'green')">Data Insights</button>
<button class="tablink" onclick="openPage('Contact', this, 'blue')">Logistic Regression ML</button>

<div id="Home" class="tabcontent">
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


### 4. Grouping the Countries into Region

There were around 42 unique entries in the country column and around 583 entries where unknow.

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
</div>













































