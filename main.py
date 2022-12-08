import argparse

def parse_me():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--arg')

    args = parser.parse_args()
    
    return args

def main():
    args = parse_me()
    
    # Generate data

    # Learning 

    # Evaluation

if __name__ == '__main__':
    main()