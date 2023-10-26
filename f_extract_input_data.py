# Funtion to extract details from "Input_ANN.txt" file

def f_extract_input(input_filename):

    """
    'input_filename' is a string - name of input file
    Input file must be present in main folder
    """
    
    print("\nReading input file");
    
    input_file_data = {}; # dictionary to store data in input file
    
    with open(input_filename) as f:
        for line in f:

            #each line stripped at '[' character
            (key, val) = line.rstrip("\n").split('['); #before'[' stored as key and after '[' stored as value
            input_file_data[key] = val.split(',');
    

    input_file_data = {a.replace('\t', ''): b for a, b in input_file_data.items()}; #tab removed from key
    input_file_data = {x.replace(' ', ''): v for x, v in input_file_data.items()}; #space removed from key

    print("Done.");
    
    return (input_file_data)