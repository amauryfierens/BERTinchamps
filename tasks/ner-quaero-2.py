# Inspired by this blog post on medium from Yanis Labrak: https://medium.com/@yanis.labrak/how-to-train-drbert-on-french-biomedical-named-entity-recognition-ner-using-huggingface-and-c84c91456d56
# (c) Amaury Fierens, UCLouvain

from datasets import load_dataset
import os
import uuid
import argparse
import itertools
import json

import cramming
import hydra
from omegaconf import OmegaConf

import evaluate
import numpy as np
from sklearn.metrics import classification_report

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy

@hydra.main(config_path="../cramming/config", config_name="cfg_medical", version_base="1.1")
def launch(cfg):

    task = "ner"
    batch_size = 8
    EPOCHS = 100

    pre_path = ""
    # Directory containing your JSON files => choices : MEDLINE_JSON, EMEA_JSON
    dir_path = pre_path+f'local_data/tmp/{cfg.quaero}_JSON/'

    train_name = "train.json"
    val_name = "dev.json"
    test_name = "test.json"
    dataset = load_dataset("json", data_files={"train":os.path.join(dir_path, train_name), "validation":os.path.join(dir_path, val_name), "test":os.path.join(dir_path, test_name)})

    train_dataset = dataset["train"]
    dev_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Get all unique labels
    label_list = set()

    for instance in train_dataset:
        label_list.update(instance["labels"])

    # Print unique labels
    print(label_list)
    label_list = list(label_list)

    def getConfig(raw_labels):

        label2id = {}
        id2label = {}

        for i, class_name in enumerate(raw_labels):
            label2id[class_name] = str(i)
            id2label[str(i)] = class_name

        return label2id, id2label

    label2id, id2label = getConfig(label_list)

    print(label2id)
    print(id2label)

    
    if cfg.model=="BERTinchamps":
        ######### BERTinchamps architecture loading #########
        tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)

        setup = cramming.utils.system_startup(cfg)
            
        thismodel = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=len(label_list), for_token_classif=True)
        model_engine, _, _, _ = cramming.load_backend(thismodel, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
        # Comment if there is no checkpoints
        model_engine.load_checkpoint(cfg_arch, model_file)
        model_engine.train()
        model = model_engine.model
        
    elif cfg.model=="camembert":
        ######### Camembert architecture loading #########
        tokenizer = AutoTokenizer.from_pretrained("camembert-base")
        model = AutoModelForTokenClassification.from_pretrained("camembert-base", num_labels=len(label_list))

    elif cfg.model=="DrBERT":
        ######### Dr-BERT architecture loading #########
        tokenizer = AutoTokenizer.from_pretrained("Dr-BERT/DrBERT-7GB")
        model = AutoModelForTokenClassification.from_pretrained("Dr-BERT/DrBERT-7GB", num_labels=len(label_list))
    
    model.config.label2id = label2id
    model.config.id2label = id2label

    def tokenize_and_align_labels(examples):

        label_all_tokens = True
        tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True, max_length=128, padding="max_length")

        labels = []

        for i, label in enumerate(examples[f"labels"]):

            label_ids = []
            previous_word_idx = None
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            for word_idx in word_ids:

                if word_idx is None:
                    label_ids.append(-100)

                elif word_idx != previous_word_idx:
                    label_ids.append(int(label2id[label[word_idx]]))  # Convert label to integer using label2id and cast to int

                else:
                    label_ids.append(int(label2id[label[word_idx]]) if label_all_tokens else -100)  # Convert label to integer using label2id and cast to int
                
                previous_word_idx = word_idx

            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels

        return tokenized_inputs


    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
    dev_tokenized_datasets = dev_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)

    if cfg.model=="BERTinchamps":
        output_name = f"BERTinchamps-QUAERO-{task}-{str(uuid.uuid4().hex)}"
    elif cfg.model=="camembert":
        output_name = f"camembert-QUAERO-{task}-{str(uuid.uuid4().hex)}"
    elif cfg.model=="DrBERT":
        output_name = f"DrBERT-QUAERO-{task}-{str(uuid.uuid4().hex)}"

    args = TrainingArguments(
        output_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        metric_for_best_model="macro_f1",
        load_best_model_at_end=True,
        greater_is_better=True,
    )

    print('Load Metrics')
    metric  = evaluate.load("seqeval", experiment_id=output_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        macro_values = [results[r]["f1"] for r in results if "overall_" not in r]
        macro_f1 = sum(macro_values) / len(macro_values)

        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"], "macro_f1": macro_f1}

    trainer = Trainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=dev_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.evaluate()


    predictions, labels, _ = trainer.predict(test_tokenized_datasets)
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    f1_score = classification_report(
        list(itertools.chain.from_iterable(true_labels)),
        list(itertools.chain.from_iterable(true_predictions)),
        digits=4,
    )
    print(f1_score)



if __name__ == "__main__":
    launch()
