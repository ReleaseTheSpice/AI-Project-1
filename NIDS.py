##############################################################
### Martin Van Laethem  - Aiden Lonquist    - Cole Atkinson ###
### A01189452           - A01189452         - A01189452     ###
###  AI Project 1 - NIDS                                    ###
###############################################################

import csv

class NIDS:
    """class for the network intrusion detection system"""

    def __init__(self):
        """Constructor"""
        self.loadFile("UNSW-NB15-BALANCED-TRAIN.csv")

    def loadFile(self, fileName):
        """Loads the csv file and... does some sort of analysis?"""
        with open(fileName, 'r') as file:   # open in read mode
            reader = csv.reader(file)
            lineCount = 0
            for row in reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                print(
                    f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
                line_count += 1
            print(f'Processed {line_count} lines.')