## Titanic: Machine Learning Competition

**Project description:** 

This is the legendary Titanic ML competition from *Kaagle.com* – the challenge was to dive into ML competitions and familiarize myself with how the Kaggle platform works. <br><br>
*The competition is to: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.*
<br><br>
The data has been split into two groups:
1. Training set 
2. Test set

### 1. Dataset Overview
```
tit.head()

   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S
```

### 2. Checking for NULL Values in the dataset
```
tit.isnull().sum()

PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```
Treating the NULL Values.

#### 1. Replacing the NULL values in the "Age" column with the **average age** of people in the ship

```
tit['Age']=tit['Age'].fillna(tit.Age.mean())
```
#### 2. Replacing the NULL values in the "Embarked" column with the value **C**.

```
Filtering the NULL Values - tit[tit['Embarked'].isnull()]

     PassengerId  Survived  Pclass  ...  Fare Cabin  Embarked
61            62         1       1  ...  80.0   B28       NaN
829          830         1       1  ...  80.0   B28       NaN

tit['Embarked'].fillna('C',inplace=True)
```
#### 3. Replacing the NULL values in the "Cabin" column with **OT (Others)**

```
tit['Cabin'].fillna('OT',inplace=True)
```
All the NULL values are treated.

```
tit.isnull().sum()

PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Cabin          0
Embarked       0
dtype: int64
```

### 3. *Cabin* Columns Analysis

Changing the values in the cabin column to an understandable format. Getting the first letter of the values and saving it in a new column.

```
tit['Cabin'].value_counts()

OT             687
B96 B98          4
C23 C25 C27      4
G6               4
D                3
C49              1
B73              1
B39              1
C99              1
C103             1
Name: Cabin, Length: 148, dtype: int64

Creating a new column.
tit['cab']=tit['Cabin'].map(lambda x: x[0])

tit['cab'].value_counts()

O    687
C     59
B     47
D     33
E     32
A     15
F     13
G      4
T      1
Name: cab, dtype: int64
```
Converting the Cabin information to reserved & un-reserved.

```
Creating a new column.
tit['cabstatus']='Reserved'
tit['cabstatus'][tit['cab'].isin(['O'])]='UnReserved'

tit['cabstatus'].value_counts()

UnReserved    687
Reserved      204
Name: cabstatus, dtype: int64
```
























