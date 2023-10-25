import csv

# Step 1: Read verb and noun CSV files and create dictionaries
verb_dict = {}
noun_dict = {}
merged_dict = {}

with open('EPIC_100_verb_classes.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        verb_dict[row['id']] = row['key']

with open('EPIC_100_noun_classes.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        noun_dict[row['id']] = row['key']

# Merge dictionaries with unique IDs
next_id = len(verb_dict)
for id, key in noun_dict.items():
    # Handle special cases where key contains a colon
    if ':' in key:
        parts = key.split(':')
        key = f"{parts[1]} {parts[0]}"
    
    merged_dict[str(next_id)] = key
    next_id += 1


# Append verb_dict to merged_dict
merged_dict.update(verb_dict)

# Step 2: Write merged labels to labels.csv
with open('epic_kitchen_100_labels.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['id', 'name'])  # Write header

    for id, label in sorted(merged_dict.items(), key=lambda x: int(x[0])):
        csv_writer.writerow([id, label])

# Step 3: Modify the existing script to write multiple labels to epic_100_test.txt
with open('EPIC_100_validation.csv', 'r') as csv_file, open('epic_100_val.txt', 'w') as txt_file:
    csv_reader = csv.DictReader(csv_file)
    
    for row in csv_reader:
        video_name = row['video_id'] + ".mp4"
        start_timestamp = row['start_timestamp']
        stop_timestamp = row['stop_timestamp']
        verb_class = row['verb_class']
        
        # Extract all_noun_classes, remove brackets, and ensure numbers are separated by spaces
        noun_classes = row['all_noun_classes'].strip('[]').replace(" ", "").split(',')
        
        # Adjust noun class IDs based on merged dictionary
        adjusted_noun_classes = [str(int(n) + len(verb_dict)) for n in noun_classes]
        
        # Write to txt file with multiple labels
        all_classes = ' '.join([verb_class] + adjusted_noun_classes)
        txt_file.write(f"{video_name} {start_timestamp} {stop_timestamp} {all_classes}\n")

print("epic_100_test.txt and labels.csv generated successfully!")
