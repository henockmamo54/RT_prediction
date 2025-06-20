import os
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


def read_n_merge(skip,file):
    """
    Reads a CSV file, optionally skipping initial rows, and performs basic parsing.

    Args:
        skip (int): Number of rows to skip at the beginning of the file (The starting row of d2ome output is inconsistent).
        file (str): Path to the CSV file.

    Returns:
        list: A list containing:
            - bool: True if parsing was successful and expected columns are present, False otherwise.
            - pd.DataFrame or None: The parsed DataFrame if successful, otherwise None.
    """
    try:
        # Read the CSV file, skipping specified rows, and disable index column creation from file.
        file_data=pd.read_csv(file,skiprows=skip,index_col=False)
        # Strip leading/trailing whitespace from column names.
        file_data.columns=[x.strip() for x in file_data.columns]
        # Define a set of expected column names for successful parsing.
        expected_cols = {'Peptide', 'UniqueToProtein', 'Exchangeable Hydrogens', 'Charge',
                         'm/z(Sequence)', 'M0', 'M1', 'M2', 'M3', 'M4'}
        # Check if all expected columns are present in the DataFrame.
        isparsed=(expected_cols.issubset(set(file_data.columns)))
        # Further ensure that the DataFrame is not empty.
        isparsed=isparsed and file_data.shape[0]>0
    except:
        isparsed=False
    if isparsed:
        # If parsing is successful, extract the protein name from the filename and add it as a new column.
        file_data['Protein']=file.split('\\')[-1].replace('.Quant.csv','')
        return [isparsed,file_data]
    else:
        # If parsing fails, return False and None.
        return [isparsed,None]

def get_df_all_quant_files(file):
    """
    Attempts to read a quantification file, trying different skip row strategies.

    This function first tries to read the file by skipping 1 row. If that fails
    (based on the criteria in `read_n_merge`), it tries again by skipping 3 rows.

    Args:
        file (str): Path to the quantification CSV file.

    Returns:
        pd.DataFrame or None: The parsed DataFrame if successful, otherwise None.
    """
    # Attempt to read the file skipping 1 row.
    res=read_n_merge(1,file)
    all_data=None
    if res[0]: # If the first attempt was successful
        all_data=res[1]
    else: # If the first attempt failed, try skipping 3 rows.
        all_data=read_n_merge(3,file)[1]
    return all_data

def getquantfile(data_path):
    """
    Reads all '.Quant.csv' files in a specified directory in parallel and concatenates them.

    Args:
        data_path (str): The path to the directory containing the quantification files.

    Returns:
        pd.DataFrame: A single DataFrame containing data from all successfully read files.
    """
    # Generate a list of full file paths for all '.Quant.csv' files in the directory.
    files_to_process = [os.path.join(data_path,x) for x in os.listdir(data_path) if '.Quant.csv' in x]
    # Process files in parallel using all available CPU cores.
    # `delayed` is used to create a callable for `get_df_all_quant_files` for each file.
    results = Parallel(n_jobs=-1)(delayed(get_df_all_quant_files)(file) for file in files_to_process)
    # Concatenate all resulting DataFrames into a single DataFrame.
    all_data=pd.concat(results)
    # Reset the index of the combined DataFrame.
    all_data=all_data.reset_index(drop=True)
    # Strip whitespace from column names in the final DataFrame.
    all_data.columns=[x.strip() for x in all_data.columns]
    return all_data

def get_rt_info(data_path):
    """
    Loads quantification data, cleans it, filters it, and extracts retention time (rt) information.

    Args:
        data_path (str): The path to the directory containing the quantification files.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Protein', 'Peptide', 'Charge',
                      'm/z(Sequence)', 'IonScore', 'rt'].
    """
    # Load and merge all quantification files from the specified path.
    data = getquantfile(data_path)

    print

    # --- Data Cleaning and Filtering ---
    # Remove rows where 'IonScore' is blank (after stripping whitespace).
    data = data[data['IonScore'].str.strip() != '']
    # Convert 'IonScore' column to float type.
    data['IonScore'] = data['IonScore'].astype(float)
    # Filter out rows with 'IonScore' less than or equal to 10.
    data = data[data['IonScore'] > 10]

    # Convert 'Start Elution (min)' and 'End Elution (min)' columns to float type.
    data['Start Elution (min)'] = data['Start Elution (min)'].astype(float)
    data['End Elution (min)'] = data['End Elution (min)'].astype(float)

    # Calculate the duration of the elution peak.
    elution_duration = data['End Elution (min)'] - data['Start Elution (min)']
    # Filter out rows where the elution duration is 1 minute or longer.
    data = data[elution_duration < 1]
    # Calculate the retention time ('rt') as the midpoint of the elution window.
    data['rt'] = data['Start Elution (min)'] + elution_duration / 2

    # Filter out rows where the calculated retention time is greater than 90 minutes.
    data = data[data['rt'] <= 90]

    # --- Final Column Selection ---
    # Define the list of columns to keep in the final DataFrame.
    columns_to_keep = [
        'Protein', 'Peptide', 'Charge', 'm/z(Sequence)',
        'IonScore', 'rt'
    ]
    # Select only the specified columns.
    data = data[columns_to_keep]

    return data

def plot_polynomial_fit(x, y, degree=3,verbose=False,x_label='Retention Time', y_label='Predicted Retention Time'):
    """
    Fits a polynomial regression model to the input data (x, y) and optionally plots the results.

    Args:
        x (np.array): Independent variable (e.g., retention times from one dataset).
        y (np.array): Dependent variable (e.g., retention times from another dataset to align to).
        degree (int, optional): Degree of the polynomial. Defaults to 3.
        verbose (bool, optional): If True, generates and shows plots of data before and after alignment. Defaults to False.
        x_label (str, optional): Label for the x-axis in plots. Defaults to 'Retention Time'.
        y_label (str, optional): Label for the y-axis in plots. Defaults to 'Predicted Retention Time'.

    Returns:
        sklearn.pipeline.Pipeline: The fitted polynomial regression model.
    """
    # Create a pipeline: first, generate polynomial features, then fit a linear regression model.
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # Fit the model. x needs to be reshaped to a 2D array for scikit-learn.
    model.fit(x.reshape(-1, 1), y)

    # If verbose mode is enabled, create and display plots.
    if verbose:
        plt.figure(figsize=(12, 6)) # Set the figure size.

        # Subplot 1: Data before alignment
        plt.subplot(1,2,1) # 1 row, 2 columns, 1st subplot
        plt.scatter(x, y, alpha=0.5,color='k') # Scatter plot of original x vs y
        plt.plot(x, x, 'b-', alpha=0.5) # Plot y=x line for reference (ideal alignment)
        plt.title("Before Alignment")
        plt.xlabel(x_label+" (minutes)")
        plt.ylabel(y_label+"(minutes)")

        # Subplot 2: Data after alignment (predicted y vs original y)
        plt.subplot(1,2,2) # 1 row, 2 columns, 2nd subplot
        y_pred = model.predict(x.reshape(-1, 1)) # Predict y values using the fitted model
        plt.scatter(y_pred,y,c='k', alpha=0.5) # Scatter plot of predicted y vs original y
        plt.plot(y_pred, y_pred, 'r-') # Plot y=x line for reference (ideal alignment after prediction)
        plt.title("After Alignment")
        plt.xlabel(x_label+" (minutes)") # Note: This label might be more accurately "Predicted " + x_label
        plt.ylabel(y_label+"(minutes)")
        plt.show() # Display the plots.

    return model
