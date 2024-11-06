import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional, Union
import statsmodels.api as sm


def analyze_group_characteristics(df: pd.DataFrame, 
                                group_col: str, 
                                characteristics: List[str], 
                                continuous_vars: Optional[List[str]] = None,
                                colors: Optional[List[str]] = None) -> Dict:
    """
    Analyze and visualize differences between groups for specified characteristics.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataset
    group_col : str
        The column name that defines the groups (e.g., 'smokes')
    characteristics : list
        List of column names to analyze
    continuous_vars : list, optional
        List of continuous variables (for appropriate statistical calculations)
        If None, will treat numeric columns as continuous
    colors : list, optional
        List of two colors for the plots. If None, uses default colors.
    
    Returns:
    --------
    dict containing balance statistics, standardized differences, and figure
    """
    if continuous_vars is None:
        continuous_vars = df[characteristics].select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if colors is None:
        colors = ['#A8A8A8', '#636363']  # Default grey colors for age
    
    # Calculate summary statistics
    stats_dict = {}
    for char in characteristics:
        if char in continuous_vars:
            stats_dict[char] = df.groupby(group_col)[char].agg(['mean', 'std']).round(3)
        else:
            # For categorical variables, calculate proportion of each category
            props = df.groupby(group_col)[char].value_counts(normalize=True).unstack()
            stats_dict[char] = props
    
    # Calculate standardized differences
    std_diffs = {}
    for char in characteristics:
        if char in continuous_vars:
            # For continuous variables
            diff = (stats_dict[char].loc[1, 'mean'] - stats_dict[char].loc[0, 'mean']) / \
                   np.sqrt((stats_dict[char].loc[1, 'std']**2 + stats_dict[char].loc[0, 'std']**2) / 2)
        else:
            # For categorical variables (using first category)
            p1 = stats_dict[char].loc[1].iloc[0]
            p0 = stats_dict[char].loc[0].iloc[0]
            diff = (p1 - p0) / np.sqrt((p1 * (1-p1) + p0 * (1-p0)) / 2)
        std_diffs[char] = diff
    
    # Create visualizations
    n_chars = len(characteristics)
    fig, axes = plt.subplots(1, n_chars, figsize=(6*n_chars, 5))
    if n_chars == 1:
        axes = [axes]
    
    for i, (char, ax) in enumerate(zip(characteristics, axes)):
        if char in continuous_vars:
            # Box plot for continuous variables using grey colors
            sns.boxplot(x=group_col, y=char, data=df, ax=ax, palette=colors)
            ax.set_title(f'{char.capitalize()} Distribution', pad=20)
        else:
            # Bar plot for categorical variables using blue for male, red for female
            props = stats_dict[char]
            props.plot(kind='bar', ax=ax, color=['#FF0000', '#2E86C1'])  # Blue for male, red for female
            ax.set_title(f'{char.capitalize()} Distribution', pad=20)
            ax.legend(title=char.capitalize(), title_fontsize=10,
                     frameon=True, fancybox=True, shadow=True)
        
        ax.set_xlabel('Smoking Status')
    
    plt.suptitle('Characteristic Balance: Smokers vs Non-Smokers', 
                 fontsize=14, y=1.05)
    plt.tight_layout()
    
    return {
        'balance_statistics': stats_dict,
        'standardized_differences': std_diffs,
        'figure': fig
    }


def plot_age_relationships(df: pd.DataFrame, 
                         outcome_var: str, 
                         predictor_var: str, 
                         colors: Tuple[str, str] = ('#636363', '#2E86C1'), 
                         figsize: Tuple[int, int] = (15, 6)) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot relationships between age groups and two variables of interest.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data with 'age' and specified outcome/predictor columns
    outcome_var : str
        Name of outcome variable column to plot
    predictor_var : str 
        Name of predictor variable column to plot
    colors : tuple
        Two colors for predictor and outcome lines
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure, pandas DataFrame
    """
    # Create a copy to avoid modifying original
    df_temp = df.copy()
    
    # Create age groups and calculate means
    age_groups = pd.qcut(df_temp['age'], q=5)
    df_temp['age_group'] = age_groups
    grouped_data = df_temp.groupby('age_group', observed=True).agg({
        predictor_var: 'mean',
        outcome_var: 'mean'
    }).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    # Plot lines
    ax.plot(range(5), grouped_data[predictor_var], marker='o', label=f'{predictor_var.capitalize()} Rate',
            color=colors[0], linewidth=2, markersize=8)
    ax.plot(range(5), grouped_data[outcome_var], marker='o', label=f'{outcome_var.capitalize()} Rate',
            color=colors[1], linewidth=2, markersize=8)
    
    # Customize plot
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{int(group.left)}-{int(group.right)}" for group in grouped_data['age_group']], 
                       rotation=45)
    ax.set_title('Age as a Confounder:\n'
                 f'Association with both {predictor_var.capitalize()} and {outcome_var.capitalize()} Rates',
                 fontsize=12, pad=15)
    ax.set_xlabel('Age Groups', fontsize=10)
    ax.set_ylabel('Rate', fontsize=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig, df_temp


def plot_gender_relationships(df: pd.DataFrame, 
                            outcome_var: str, 
                            predictor_var: str, 
                            colors: Tuple[str, str] = ('#636363', '#2E86C1'), 
                            figsize: Tuple[int, int] = (15, 6)) -> Figure:
    """
    Plot relationships between gender and two variables of interest.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data with 'gender' and specified outcome/predictor columns
    outcome_var : str
        Name of outcome variable column to plot
    predictor_var : str
        Name of predictor variable column to plot
    colors : tuple
        Two colors for predictor and outcome bars
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Calculate statistics by gender
    gender_stats = df.groupby('gender', observed=True).agg({
        predictor_var: 'mean',
        outcome_var: 'mean'
    }).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    # Plot bars
    x = np.arange(len(gender_stats['gender']))
    width = 0.3
    ax.bar(x - width/2, gender_stats[predictor_var], width, label=f'{predictor_var.capitalize()} Rate',
           color=colors[0], alpha=0.8)
    ax.bar(x + width/2, gender_stats[outcome_var], width, label=f'{outcome_var.capitalize()} Rate',
           color=colors[1], alpha=0.8)
    
    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(gender_stats['gender'], fontsize=10)
    ax.set_title('Gender as a Confounder:\n'
                 f'Association with both {predictor_var.capitalize()} and {outcome_var.capitalize()} Rates',
                 fontsize=12, pad=15)
    ax.set_ylabel('Rate', fontsize=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df: pd.DataFrame, 
                          outcome_var: str, 
                          predictor_var: str, 
                          cmap: str = 'coolwarm', 
                          figsize: Tuple[int, int] = (8, 6)) -> Tuple[Figure, pd.DataFrame]:
    """
    Plot correlation matrix highlighting relationships between variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data with relevant columns
    outcome_var : str
        Name of outcome variable column
    predictor_var : str
        Name of predictor variable column
    cmap : str
        Colormap for heatmap
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure, pandas DataFrame
    """
    # Create a copy and add gender_numeric
    df_temp = df.copy()
    df_temp['gender_numeric'] = df_temp['gender'].cat.codes
    
    # Calculate correlations
    corr_vars = ['age', 'gender_numeric', predictor_var, outcome_var]
    corr_matrix = df_temp[corr_vars].corr()
    
    # Create highlight mask
    highlight_mask = np.zeros_like(corr_matrix, dtype=bool)
    highlight_mask[0, 2:] = True
    highlight_mask[2:, 0] = True
    highlight_mask[1, 2:] = True
    highlight_mask[2:, 1] = True
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                xticklabels=['Age', 'Gender', predictor_var.capitalize(), outcome_var.capitalize()],
                yticklabels=['Age', 'Gender', predictor_var.capitalize(), outcome_var.capitalize()],
                cbar_kws={'label': 'Correlation Coefficient'})
    
    # Enhance highlighted correlations
    for i, j in zip(*np.where(highlight_mask)):
        text = plt.gca().texts[i * len(corr_matrix) + j]
        text.set_weight('bold')
        text.set_size(11)
        text.set_color('black')
    
    plt.title('Correlation Matrix\nHighlighting Key Relationships',
              fontsize=12, pad=15)
    plt.tight_layout()
    return fig, df_temp

def run_logistic_regressions(df: pd.DataFrame, 
                            outcome: str,
                            predictor: str, 
                            control_sets: list[list[str]],
                            model_names: list[str] = None) -> None:
    """
    Run multiple logistic regression models with different sets of control variables
    and visualize the results.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    outcome : str 
        Name of outcome variable column
    predictor : str
        Name of main predictor variable column
    control_sets : list of lists
        List where each element is a list of control variable names for each model
    model_names : list of str, optional
        Names for each model. If None, will use "Model 1", "Model 2", etc.
    """
    # Create clean dataset and male indicator if gender is used
    df_clean = df.copy()
    if any('gender' in controls for controls in control_sets):
        df_clean['male'] = (df_clean['gender'] == 'male').astype(int)
    
    # Store results for each model
    model_results = []
    
    # Run models
    for i, controls in enumerate(control_sets):
        # Prepare features
        features = [predictor] + controls
        # Replace 'gender' with 'male' in features if present
        features = ['male' if x == 'gender' else x for x in features]
        
        # Drop rows with missing values for required variables
        required_cols = features + [outcome]
        df_model = df_clean.dropna(subset=required_cols)
        
        # Create feature matrix
        X = df_model[features]
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.Logit(df_model[outcome], X)
        results = model.fit(disp=0)
        model_results.append(results)
        
        # Print summary
        print(f"\nModel {i+1}: {outcome} ~ {predictor} + {' + '.join(controls)}")
        print(results.summary().tables[1])
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Get odds ratios and CIs for main predictor
    ors = []
    ci_lowers = []
    ci_uppers = []
    
    for results in model_results:
        or_val = np.exp(results.params[predictor])
        ci = np.exp(results.conf_int().loc[predictor])
        ors.append(or_val)
        ci_lowers.append(ci[0])
        ci_uppers.append(ci[1])
    
    # Plot
    n_models = len(model_results)
    y_pos = range(n_models)
    
    if model_names is None:
        model_names = [f"Model {i+1}\n({predictor} + {', '.join(controls)})" 
                      for i, controls in enumerate(control_sets)]
    
    plt.barh(y_pos, ors, color=plt.cm.Pastel1(np.linspace(0, 1, n_models)), height=0.4)
    
    # Add error bars
    plt.errorbar(ors, y_pos,
                xerr=[[or_val - ci_lower for or_val, ci_lower in zip(ors, ci_lowers)],
                      [ci_upper - or_val for or_val, ci_upper in zip(ors, ci_uppers)]],
                fmt='none', color='black', capsize=5)
    
    # Add reference line
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    
    plt.title(f'Comparison of {predictor.capitalize()} Effect on {outcome.capitalize()}\nAcross Different Model Specifications')
    plt.xlabel(f'Odds Ratio for {predictor.capitalize()} (with 95% CI)')
    plt.yticks(y_pos, model_names)
    plt.grid(axis='x', alpha=0.3)
    
    # Add text annotations
    for i, (or_val, ci_lower, ci_upper) in enumerate(zip(ors, ci_lowers, ci_uppers)):
        plt.text(or_val + 0.1, i, 
                f'OR = {or_val:.3f}\nCI: [{ci_lower:.3f}, {ci_upper:.3f}]',
                va='center')
    
    plt.tight_layout()
    plt.show()