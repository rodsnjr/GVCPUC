import argparse
import os

def parse():
    " Parse the args to launch this app "
    parser = argparse.ArgumentParser(description='Read images from and save the labels on csv.')

    parser.add_argument('labels',nargs='+' ,metavar='N', type=str, help='Path of the label files')

    args = parser.parse_args()
    return args

def get_file(file, file2):
    for line1 in file2:
        _file2 = line1[0]
        if _file2 == _file:
            return line1

def compare_lines(line, compare):

    print(line, compare)

    left1, left2 = line[1], compare[1]
    center1, center2 = line[2], compare[2]
    right1, right2 = line[3], compare[3]
    
    # print(left1, left2)

    mean = 0
    mean += 1 if left1 == left2 else 0
    mean += 1 if center1 == center2 else 0
    mean += 1 if right1 == right2 else 0 
    mean = mean / 3

    mean_file = { 'file' : line[0], 'mean' : mean }

    # print(mean_file)
    # print(mean_file)
    return mean_file

def overall_mean(mean_files):
    total = len(mean_files)
    sumof = 0
    for mean in mean_files:
        sumof += mean['mean']
    return sumof / total

def read_lines(path):
    file = open(os.path.join(path) ,"r")
    lines = file.readlines()
    # print(type(lines))
    file.close()

    _lines = []
    for line in lines:
        # print(line.split(','))
        _lines.append(line.split(','))

    return _lines

if __name__ == "__main__":
    args = parse()

    if len(args.labels) != 2:
        raise Exception("Must pass two csvs")
    
    print("reading files: ", args.labels[0], args.labels[1])
    file1 = read_lines(args.labels[0])
    file2 = read_lines(args.labels[1])
    mean_files = []

    for line, line1 in zip(file1, file2):
        mean_files.append(compare_lines(line, line1))

    print(overall_mean(mean_files))