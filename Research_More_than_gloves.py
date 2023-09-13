
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
# SciKit Learn Models
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, learning_curve
import warnings

warnings.filterwarnings("ignore")

############# VERİYİ OKUMA ##################################
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv(r"C:\Users\user\Desktop\data.csv", sep=',')
df = df_.copy()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe().T)


check_df(df)

############### FAZLALIK KOLON DÜŞÜRME ###############################

df.columns
opp_include_columns = []

for col in df:
    if "_opp_" in col:
        opp_include_columns.append(col)

df.drop(opp_include_columns, axis=1, inplace=True)

###### Draw Drop #####
df = df[df['Winner'] != 'Draw']
df['Winner'].value_counts()

########## WİNNER MAP  #####

df['Winner'] = df['Winner'].map({'Red': 0, 'Blue': 1})

df.head()


def grab_col_names(dataframe, cat_th=5, car_th=30):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'cat_cols: {(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'num_cols: {(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'cat_but_car: {(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'num_but_cat: {(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols


cat_cols, cat_but_car, num_cols = grab_col_names(df)


######################################
#  Analysis of Categorical Variables
######################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, False)


######################################
#  Analysis of Numerical Variables
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

    print("#####################################")


for col in num_cols:
    num_summary(df, col)


######################################
# Analysis of Target Variable
######################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Winner", col)


df["Winner"].hist(bins=100)
plt.show(block=True)



def outlier_thresholds(dataframe, variable, low_quantile=0.01, up_quantile=0.99):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit



def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "Winner":
        print(col, check_outlier(df, col))



def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "Winner":
        print(replace_with_thresholds(df, col))


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df, na_name=True)
df.isnull().sum()

########   NULL DOLDURMA   #####################

for i in num_cols:
    df[i] = df[i].fillna(df[i].mean())

from statistics import mode

df['B_Stance'] = df['B_Stance'].fillna(df['B_Stance'].mode()[0])
df['R_Stance'] = df['R_Stance'].fillna(df['R_Stance'].mode()[0])
df.isnull().sum()


df = df.drop(columns=['R_fighter', 'B_fighter', 'Referee', 'location', 'date', 'title_bout'])


################## ENCODİNG ###################


def encode_columns(df, columns):
    '''
    Veri çerçevesinde belirtilen sütunları LabelEncoder ile kodlar.

    Parametreler:
    df (pandas.DataFrame): Kodlama yapılacak veri çerçevesi.
    columns (list): Kodlanacak sütun adlarını içeren bir liste.

    Dönüş:
    pandas.DataFrame: Kodlanmış veri çerçevesi.
    '''
    enc = LabelEncoder()
    encoded_df = df.copy()

    for column in columns:
        encoded_column = enc.fit_transform(df[column])
        encoded_df[column] = encoded_column

    return encoded_df


df = encode_columns(df, ['weight_class', 'R_Stance', 'B_Stance'])

print(df.head(5))


############## SCALE  #######################

def standard_scale_columns(df, columns_to_scale):
    std = StandardScaler()
    df_to_scale = df[columns_to_scale]
    df_scaled = std.fit_transform(df_to_scale)
    df[df_to_scale.columns] = df_scaled
    return df


numerical = df.drop(['weight_class', 'Winner', 'R_Stance', 'B_Stance'], axis=1)
columns_to_scale = numerical.select_dtypes(include=[np.float64, np.int64]).columns
df = standard_scale_columns(df, columns_to_scale)

################# MODEL  ###################
X = df.drop("Winner", axis=1)
y = df["Winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)


###### MODEL FUNCTIONS #####


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                    ('KNN', KNeighborsClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


base_models(X_train, y_train, scoring="roc_auc")

LR = LogisticRegression()
LR.get_params()

KNN = KNeighborsClassifier()
KNN.get_params()

RF = RandomForestClassifier()
RF.get_params()

XGBoost = XGBClassifier(random_state=17, use_label_encoder=False)
XGBoost.get_params()




lr_params = {'max_iter': [100, 50, 200],
             'intercept_scaling': [1, 10, 50]
             }

knn_params = {"n_neighbors": range(2, 50)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("LR", LogisticRegression(), lr_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


A = hyperparameter_optimization(X_train, y_train, cv=3, scoring="roc_auc")


# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[  # ('KNN', best_models["KNN"]),
        # ('RF', best_models["RF"]),
        # ('LR', best_models["LR"]),
        ('XGBoost', best_models["XGBoost"])],

        voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


voting_classifier(A, X_test, y_test)