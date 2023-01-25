# Quick and dirty script to convert a csv file to a json lines file that can be used to train a model in google's vertex AI
# Assumes the input file is input.csv and that the csv file has the full text content in the second column, with potential annotation strings
# in the 2nd, 3rd, and 4th column.  Because it takes an annotation string, this script will always annotate the first instance of that string in the text
# Also only supports one annotation of each type per text/row at this point  

import csv
import json


def buildAnnotation(text, attributeValue, name):
    start_offset = text.find(attributeValue)
    return {"start_offset": start_offset, "end_offset": start_offset + len(attributeValue), "display_name": name}


with open('input.csv') as csv_file:
    with open('output.jsonl', 'w') as out_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        headers = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                headers = row
                line_count += 1
            else:
                text = row[1]
                action_attribute = row[2]
                text_attribute = row[3]
                element_type_attribute = row[4]
                position_attribute = row[5] 
                line_count += 1

                if text:
                    annotations = []
                    if row[3]:
                        annotations.append(buildAnnotation(text, row[3], headers[3])) 
                    if row[4]:
                        annotations.append(buildAnnotation(text, row[4], headers[4]))
                    if row[5]:
                        annotations.append(buildAnnotation(text, row[5], headers[5]))  
                    lineOut = {"textContent": text, "textSegmentAnnotations": annotations}
                    out_file.write(json.dumps(lineOut))
                    out_file.write('\n')

