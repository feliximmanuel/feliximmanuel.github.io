## Adult Dataset & their Income Analysis 

**Project description:** 

The "Adult Dataset has around 32,000 records with various information like age, education, marital-status, occupation, gender, hours per week, country and income information". From this dataset I have derived various insights and explained them below.

### Dataset Overview

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

### Checking for NULL values in the data
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


### Changing the column Names

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


### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
