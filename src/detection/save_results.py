import pandas as pd
import glob
import json
import os
from os.path import basename




def save_results_in_csv(folder_path):
    # Assuming df_f1 is already defined
    df_f1 = pd.DataFrame(columns=["Chick", "Algorithm", "F1-measure", "Precision", "Recall"])

    chick_folder = basename(folder_path)

    # Iterate over the JSON files in the folder
    for filename in glob.glob(f'{folder_path}/**/*.json', recursive=True):
        if filename.endswith("_evaluation_results.json"):
            # Read the JSON file
            with open(filename, 'r') as file:
                json_data = json.load(file)

            # Extract relevant information from the JSON structure
            chick_name = json_data.get("audiofilename", "N/A")
            algorithm = json_data.get("Algorithm", "N/A")
            f_measure = json_data.get("F-measure", "N/A")
            precision = json_data.get("Precision", "N/A")
            recall = json_data.get("Recall", "N/A")

            # Add a new row to the DataFrame
            df_f1.loc[len(df_f1)] = [chick_name, algorithm, f_measure, precision, recall]

    print(df_f1)

    csv_filename = os.path.join(folder_path, f"Overall_chicks_results_{chick_folder}.csv")

    # Save the DataFrame as a CSV file
    individual_performances_csv= df_f1.to_csv(csv_filename, index=False)

    print("Overall individual performances saved in .csv:", csv_filename)

    return individual_performances_csv







############################### Convert in Latex table #########################################

def save_global_results_latex(data_folder):
    # Import the JSON file from the folder
    with open(data_folder + '\\global_evaluation_results.json') as json_file:
        data = json.load(json_file)

    # Create a DataFrame from the transposed dictionary
    df = pd.DataFrame(data).transpose()

    folder_name = os.path.basename(data_folder)

    # output_csv_filename= f'evaluation_results_{folder_name}.csv'
    # output_latex_filename = f'evaluation_results_{folder_name}.tex'

    output_csv_filename = os.path.join(data_folder, f'evaluation_results_{folder_name}.csv')
    output_latex_filename = os.path.join(data_folder, f'evaluation_results_{folder_name}.tex')


    # Save the DataFrame as a CSV file
    table_csv = df.to_csv(output_csv_filename, index_label='Method')
    
    # Create a subset of the DataFrame for the LaTeX table
    df_subset = df.iloc[:, :4]

    # Create the LaTeX table
    latex_table = df_subset.to_latex(index=True, escape=False)

    # Save the LaTeX table as a .tex file
    with open(output_latex_filename, 'w') as latex_file:
        latex_file.write(latex_table)


    return table_csv, latex_table


################################### Save the results in a .csv file distinguishing for sexes and groups ########################################



def save_detected_calls_in_csv(folder_path):
    # Define the DataFrame columns
    df_columns = ["Chick", "Algorithm", "Numbers_of_calls", "Group", "Sex"]

    # Create an empty DataFrame with defined columns
    df_f1 = pd.DataFrame(columns=df_columns)

    chick_folder = basename(folder_path)

    # Iterate over the JSON files in the folder
    for filename in glob.glob(f'{folder_path}/**/*.json', recursive=True):
        if filename.endswith("_calls_detected_.json"):
            # Read the JSON file
            with open(filename, 'r') as file:
                json_data = json.load(file)

            # Extract relevant information from the JSON structure
            chick_name = json_data.get("audiofilename", "N/A")
            algorithm = json_data.get("Algorithm", "N/A")
            numb_calls = json_data.get("Number of calls", "N/A")
            group = json_data.get("Group", "N/A")
            sex = json_data.get("Sex", "N/A")

            # Create a list with values to append to DataFrame
            row_values = [chick_name, algorithm, numb_calls, group, sex]

            # Add a new row to the DataFrame
            df_f1.loc[len(df_f1)] = row_values

    csv_filename = os.path.join(folder_path, f"Total_number_of_calls_detected_{chick_folder}.csv")

    # Save the DataFrame as a CSV file
    df_f1.to_csv(csv_filename, index=False)

    print("Total_number_of_calls_detected.csv:", csv_filename)

    return csv_filename