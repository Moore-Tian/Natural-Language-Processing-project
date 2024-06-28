import csv

def convert_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as input_csv, open(output_file, 'w', encoding='utf-8', newline='') as output_csv:
        reader = csv.DictReader(input_csv)
        fieldnames = ['word', 'expected']
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            word = row['word']
            writer.writerow({'word': word, 'expected': 'O'})

# 使用示例
input_file = 'data/test.csv'
output_file = 'data/conver_test.csv'

convert_csv(input_file, output_file)
print("Conversion complete.")