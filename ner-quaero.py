import os
import json
import spacy

# Initialize a spaCy tokenizer
nlp = spacy.blank('fr')
nlp2 = spacy.blank('fr')
nlp2.add_pipe("sentencizer")


def brat_to_json_medline(data_dir, output_file):
    data = []
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        # Load text file
        with open(os.path.join(data_dir, txt_file), 'r') as f:
            text = f.read()
        
        # Tokenize text
        doc = nlp(text)
        tokens = [token.text for token in doc]
        
        # Load corresponding annotation file
        ann_file = txt_file.replace('.txt', '.ann')
        with open(os.path.join(data_dir, ann_file), 'r') as f:
            annotations = f.readlines()
        
        # Process annotations
        entities = []
        for ann in annotations:
            parts = ann.strip().split('\t')
            if parts[0].startswith('T'):
                entity_info = parts[1].split(' ')
                label = "MED" # TODO: entity_info[0] if NER, "MED" otherwise
                spans = ' '.join(entity_info[1:]).split(';')  # Get spans, excluding label and entity text
                # Handle the case of multiple spans for the same entity
                for span in spans:
                    start, end = map(int, span.split())  # Convert to int
                    entity = {'entity': parts[0], 'label': label, 'start': start, 'end': end, 'text': parts[2]}
                    entities.append(entity)

        # Assign labels to tokens
        labels = []
        for token in doc:
            label = '0'
            for entity in entities:
                if entity['start'] <= token.idx < entity['end']:
                    label = entity['label']
                    break
            labels.append(label)

        # Add to data
        data.append({'tokens': tokens, 'labels': labels})
    
    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def brat_to_json_emea(data_dir, output_file):
    data = []
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        # Load text file
        with open(os.path.join(data_dir, txt_file), 'r') as f:
            text = f.read()
        
        # Split text into sentences
        doc = nlp2(text)
        sentences = list(doc.sents)
        
        # Load corresponding annotation file
        ann_file = txt_file.replace('.txt', '.ann')
        with open(os.path.join(data_dir, ann_file), 'r') as f:
            annotations = f.readlines()
        
        # Process annotations
        entities = []
        for ann in annotations:
            parts = ann.strip().split('\t')
            if parts[0].startswith('T'):
                entity_info = parts[1].split(' ')
                label = "MED" # TODO: = entity_info[0] if NER, MED otherwise
                spans = ' '.join(entity_info[1:]).split(';')  # Get spans, excluding label and entity text
                # Handle the case of multiple spans for the same entity
                for span in spans:
                    start, end = map(int, span.split())  # Convert to int
                    entity = {'entity': parts[0], 'label': label, 'start': start, 'end': end, 'text': parts[2]}
                    entities.append(entity)

        # Assign labels to tokens
        for sent in sentences:
            tokens = [token.text for token in sent]
            labels = []
            for token in sent:
                label = '0'
                for entity in entities:
                    if entity['start'] <= token.idx < entity['end']:
                        label = entity['label']
                        break
                labels.append(label)

            # Add to data
            # Ensure that each sentence is added as a separate item in the data list
            data.append({'tokens': tokens, 'labels': labels})
    
    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Paths to your folders
train_dir = '/globalscratch/ucl/ingi/amfieren/Datasets/QUAERO/QUAERO_FrenchMed/corpus/train/'
dev_dir = '/globalscratch/ucl/ingi/amfieren/Datasets/QUAERO/QUAERO_FrenchMed/corpus/dev/'
test_dir = '/globalscratch/ucl/ingi/amfieren/Datasets/QUAERO/QUAERO_FrenchMed/corpus/test/'

# Convert and save as JSON
brat_to_json_medline(train_dir+"MEDLINE", '/globalscratch/ucl/ingi/amfieren/Datasets/QUAERO/MEDLINE_JSON/only_med_train.json')
brat_to_json_medline(dev_dir+"MEDLINE", '/globalscratch/ucl/ingi/amfieren/Datasets/QUAERO/MEDLINE_JSON/only_med_dev.json')
brat_to_json_medline(test_dir+"MEDLINE", '/globalscratch/ucl/ingi/amfieren/Datasets/QUAERO/MEDLINE_JSON/only_med_test.json')

brat_to_json_emea(train_dir+"EMEA", '/globalscratch/ucl/ingi/amfieren/Datasets/QUAERO/EMEA_JSON/only_med_train.json')
brat_to_json_emea(dev_dir+"EMEA", '/globalscratch/ucl/ingi/amfieren/Datasets/QUAERO/EMEA_JSON/only_med_dev.json')
brat_to_json_emea(test_dir+"EMEA", '/globalscratch/ucl/ingi/amfieren/Datasets/QUAERO/EMEA_JSON/only_med_test.json')
