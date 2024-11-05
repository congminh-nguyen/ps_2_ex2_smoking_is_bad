import polars as pl
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm

#clean the data
def data_clean(df):
    """""
    Clean the data by converting the columns to the appropriate data types.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    """

    #change smoker and outcome to binary
    df['smoker'] = df['smoker'].map({"Yes": 1, "No": 0})
    df['outcome'] = df['outcome'].map({"Alive": 1, "Dead": 0})

    #change age, outcome, smoker to numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce')
    df['smoker'] = pd.to_numeric(df['smoker'], errors='coerce')

    #change gender to categorical
    df['gender'] = df['gender'].astype('category')

    return df

#plots
def visualize_data(df):
    """""
    Visualize the data using a pairplot and a boxplot.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    """""

    plt.figure(figsize=(8, 6))

    plt.subplot(1, 2, 1)
    sns.countplot(x='gender', data=df)
    plt.title('Alive vs Gender')

    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='age', hue='alive', multiple='stack', bins=30)
    plt.title('Age Distribution by ALive Status')

    plt.tight_layout()
    plt.show()

def plot_alive_probability_by_age(df, alive_col='alive', age_col='age'):
    """
    Plot the mean probability of being alive by age.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    alive_col (str): The name of the column indicating alive status (binary).
    age_col (str): The name of the column for age.
    """
    # Group by age and calculate the mean probability of being alive
    mean_prob_by_age = df.groupby(age_col).agg(prob=(alive_col, np.mean)).reset_index()

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=mean_prob_by_age, x=age_col, y='prob', marker='o')

    # Add titles and labels
    plt.title('Mean Probability of Being Alive by Age')
    plt.xlabel('Age')
    plt.ylabel('Mean Probability of Being Alive')
    plt.ylim(0, 1)  # Since probability ranges from 0 to 1
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_smoker_probability_by_age(df, smokes_col='smokes', age_col='age'):
    """
    Plot the mean probability of smoking by age.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    alive_col (str): The name of the column indicating alive status (binary).
    age_col (str): The name of the column for age.
    """
    # Group by age and calculate the mean probability of being alive
    mean_prob_by_age = df.groupby(age_col).agg(prob=(smokes_col, np.mean)).reset_index()

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=mean_prob_by_age, x=age_col, y='prob', marker='o')

    # Add titles and labels
    plt.title('Mean Probability of Smoking by Age')
    plt.xlabel('Age')
    plt.ylabel('Mean Probability of Smoking')
    plt.ylim(0, 1)  # Since probability ranges from 0 to 1
    plt.grid(True)

    # Show the plot
    plt.show()

def logistic_reg(df):
    X = df[['age', 'smokes']]

    X = pd.get_dummies(X, drop_first=True)
    y = df['alive']

    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit()

    return result.summary

def fit_smoking_model(df):
    X = df[['age']]
    y = df['smokes']

    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    return result

def smoking_liklihood(model, ages):
    age_df = pd.DataFrame({'age': ages})
    age_df = sm.add_constant(age_df)

    probabilities = model.predict(age_df)

    return pd.DataFrame({'age': ages, 'probability': probabilities})

def fit_survival_model(df, features, target_col='alive'):
    """
    Fit a logistic regression model to predict survival.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    features (list): List of feature column names.
    target_col (str): The name of the column indicating survival status.

    Returns:
    result: Fitted logistic regression model.
    """
    # Prepare the features and target variable
    X = df[features]
    y = df[target_col]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit logistic regression model
    model = sm.Logit(y, X)
    result = model.fit(disp=0)  # Suppress output

    return result

def plot_variable_contributions(model, features):
    """
    Plot the contributions of variables to survival based on the model coefficients.

    Parameters:
    model: Fitted logistic regression model.
    features (list): List of feature column names.
    """
    # Get coefficients and their corresponding feature names
    coeffs = model.params[1:]  # Exclude the intercept
    conf = model.conf_int().iloc[1:]  # Get confidence intervals for the features
    conf['OR'] = coeffs.values
    conf.columns = ['2.5%', '97.5%', 'OR']

    # Calculate odds ratios and their confidence intervals
    conf['OR'] = np.exp(conf['OR'])  # Exponentiate to get odds ratios
    conf['2.5%'] = np.exp(conf['2.5%'])
    conf['97.5%'] = np.exp(conf['97.5%'])

    # Plot the odds ratios with confidence intervals
    plt.figure(figsize=(10, 6))
    sns.barplot(x=conf['OR'], y=conf.index, palette='viridis', capsize=.2)

    # Add confidence intervals as error bars
    plt.errorbar(conf['OR'], conf.index,
                 xerr=[conf['OR'] - conf['2.5%'], conf['97.5%'] - conf['OR']],
                 fmt='none', c='black', capsize=5)

    plt.title('Variable Contributions to Survival')
    plt.xlabel('Odds Ratio (OR)')
    plt.ylabel('Variables')
    plt.axvline(1, linestyle='--', color='red')  # Line for odds ratio of 1
    plt.grid(axis='x')

    # Show the plot
    plt.show()
