# Application Model for CSV Files
# Load and Save Dataset CSV Files
import csv

class CSVModel:
    def __init__(self, file):
        self.file = file
        self.items = []
    
    def add(self, _file, labels):
        self.items.append([_file, labels[0], labels[1], labels[2]])
    
    def last(self):
        return self.items.pop()
    
    def load(self):
        with open(self.file, newline='') as csvfile:
            lines = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for line in lines:
                self.items.append(line)

    def save(self):
        with open(self.file, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for val in self.items:
                spamwriter.writerow(val)