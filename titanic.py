# 导入警告
import warnings
warnings.filterwarnings('ignore')

# 数据处理清洗包
import pandas as pd
import numpy as np

# 机器学习算法相关包
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 加载数据集
# 使用绝对路径加载数据集
train_df = pd.read_csv('H:/develop/vscodeprogram/kaggle program/titanic/train.csv')
test_df = pd.read_csv('H:/develop/vscodeprogram/kaggle program/titanic/test.csv')
combine = [train_df, test_df]

# 打印数据集的基本信息
print(train_df.columns.values)
print(train_df.head())
print(train_df.tail())
print(train_df.isnull().sum())
train_df.info()
print('_' * 40)

# 分析几个因素与幸存情况之间的相关性
print(round(train_df.describe(percentiles=[.5, .6, .7, .75, .8, .9, .99]), 2))
print(train_df[['Pclass', 'Survived']]
      .groupby(['Pclass'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))
print(train_df[['Sex', 'Survived']]
      .groupby(['Sex'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))
print(train_df[['SibSp', 'Survived']]
      .groupby(['SibSp'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))
print(train_df[['Parch', 'Survived']]
      .groupby(['Parch'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))

# 删除无用特征 Ticket 和 Cabin
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# 从现有特征中提取新特征
print(train_df['Name'].head(10))

# 使用正则表达式提取特征 Title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# 检查 Title 和 Sex 的交叉表
print(pd.crosstab(train_df['Title'], train_df['Sex']).sort_values(by='female', ascending=False))

# 用常见的标题将许多标题替换为稀有
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')

# 打印替换后的 Title 与幸存情况的统计
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 定义 Title 的映射关系
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

# 将 Title 映射为数值
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)  # 填充未映射的值为 0

# 查看处理后的数据
print(train_df.head())

# 删除训练集中的 Name 特征和测试集中的 PassengerId 特征
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

# 转换性别特征 Sex
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

print(train_df.head())

# 创建空数组
guess_ages = np.zeros((2, 3))

# 遍历 Sex（0或1）和 Pclass（1，2，3）来计算六种组合的 Age 猜测值
for dataset in combine:
    # 第一个 for 循环计算每一个分组的 Age 预测值
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()
            age_guess = guess_df.median()
            # 将随机年龄浮点数转换为最接近的 0.5 年龄（四舍五入）
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    # 第二个 for 循环对空值进行赋值
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == i) & (dataset['Pclass'] == j + 1),
                        'Age'] = guess_ages[i, j]
    dataset['Age'] = dataset['Age'].astype(int)

# 将年龄分割为 5 段，等距分箱
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

# 将年龄分段
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# 删除训练集中的 AgeBand 特征
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

# 创建新特征 FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 创建新特征 IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# 舍弃 Parch、SibSp 和 FamilySize 特征，转而支持 IsAlone
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

print(train_df.head())

# 创建新特征 Age*Pclass
for dataset in combine:
    dataset['Age*Pclass'] = dataset['Age'] * dataset['Pclass']

print(train_df.loc[:, ['Age*Pclass', 'Age', 'Pclass']].head(10))
print(train_df[['Age*Pclass', 'Survived']].groupby(['Age*Pclass'], as_index=False).mean())

# 填充 Embarked 缺失值
freq_port = train_df['Embarked'].dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 将 Embarked 映射为数值
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 填充 Fare 缺失值
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# 分离特征和目标变量
X = train_df.drop('Survived', axis=1)  # 特征
Y = train_df['Survived']  # 目标变量

# 测试集特征
X_test = test_df.drop('PassengerId', axis=1)  # 假设测试集不包含目标变量

# 初始化随机森林模型
random_forest = RandomForestClassifier(n_estimators=100)

# 训练随机森林模型
random_forest.fit(X, Y)

# 在测试集上进行预测
Y_pred = random_forest.predict(X_test)

# 生成提交文件
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})

submission.to_csv('H:\\develop\\vscodeprogram\\kaggle program\\titanic_random_forest_submission.csv', index=False)