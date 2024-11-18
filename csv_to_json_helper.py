import csv
import json

# Specify the file paths
csv_file_path = 'singlish_to_english_v0.4.csv'
json_file_path = 'singlish_to_english_v0.4.json'

# Open the CSV file and read its contents
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # Convert rows into a list of dictionaries
    data = [row for row in csv_reader]

# Write the JSON file
with open(json_file_path, mode='w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4)

print(f"CSV data has been successfully converted to JSON and saved to {json_file_path}")
