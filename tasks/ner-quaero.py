# (c) Amaury Fierens, UCLouvain
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
                label = entity_info[0] # TODO: entity_info[0] if NER, "MED" otherwise
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
    with open(output_file, 'w+') as f:
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
                label = entity_info[0] # TODO: = entity_info[0] if NER, MED otherwise
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
    with open(output_file, 'w+') as f:
        json.dump(data, f, indent=4)

# Paths to your folders
pre = ""
data_path = "local_data/QUAEROFrenchMedCorpus/"
train_dir = pre+data_path+'train/'
dev_dir = pre+data_path+'dev/'
test_dir = pre+data_path+'test/'

data_path = "local_data/tmp/MEDLINE_JSON/"
# Convert and save as JSON
brat_to_json_medline(train_dir+"MEDLINE", pre+data_path+'train.json')
brat_to_json_medline(dev_dir+"MEDLINE", pre+data_path+'dev.json')
brat_to_json_medline(test_dir+"MEDLINE", pre+data_path+'test.json')

data_path = "local_data/tmp/EMEA_JSON/"
brat_to_json_emea(train_dir+"EMEA", pre+data_path+'train.json')
brat_to_json_emea(dev_dir+"EMEA", pre+data_path+'dev.json')
brat_to_json_emea(test_dir+"EMEA", pre+data_path+'test.json')
