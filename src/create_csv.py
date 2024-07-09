import pandas as pd
import json

def create_csv(synthetic_therapies, output_file):
    data = {
        'instruction': ["According the text in column 'Input', classify with YES or NO the immune deficiency!"] * len(synthetic_therapies),
        'input': [therapy for therapy, _ in synthetic_therapies],
        'output': [label for _, label in synthetic_therapies]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    with open('../data/synthetic_therapies.json', 'r') as f:
        synthetic_therapies = json.load(f)
    
    create_csv(synthetic_therapies, '../data/therapy_classification.csv')
