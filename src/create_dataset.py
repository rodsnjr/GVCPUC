from general.application.launch import launch
import argparse
# Rodar o cliente que lê um diretório e guarda o CSV do Dataset (deste diretório)
# Teclas:
# 1, 2,3 (labels)
# Q - Save and Close
# S - Save
# N, P (Next/Previous)

def parse():
    " Parse the args to launch this app "
    parser = argparse.ArgumentParser(description='Read images from and save the labels on csv.')
    
    parser.add_argument('--dir', type=str, help='Path of the images')
    parser.add_argument('--output', type=str, help="Path of the csv")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    launch(args.dir, args.output)