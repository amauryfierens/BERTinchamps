from datasets import load_dataset
import os
import uuid
import itertools
import json

import cramming
import hydra
from omegaconf import OmegaConf

import evaluate
import numpy as np
from sklearn.metrics import classification_report

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, PreTrainedTokenizerFast

@hydra.main(config_path="../cramming/config", config_name="cfg_rtbf", version_base="1.1")
def launch(cfg):
    task = "seq_classif"
    batch_size = 32
    EPOCHS = 20
    
    # Directory containing your JSON file => replace the TODO with the path to your data
    pre_path = "/TODO/BERTinchamps/"
    dir_path = 'local_data/Testing_corpora/'
    
    ks = cfg.nbrs
    if cfg.classic==False:
        tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
        setup = cramming.utils.system_startup(cfg)
        
    for k in ks:
        if cfg.is_signature:
            train_name = f"rtbfCorpus_{k}_train_signature.json"
            test_name = f"rtbfCorpus_{k}_test_signature.json"
        else:
            train_name = f"rtbfCorpus_{k}_train.json"
            test_name = f"rtbfCorpus_{k}_test.json"
        dataset = load_dataset("json", data_files={"train":os.path.join(pre_path+dir_path, train_name), "test":os.path.join(pre_path+dir_path, test_name)})
        
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        
        label_list = set()
        
        for instance in train_dataset:
            if cfg.is_signature:
                label_list.add(instance["signature"])
            else:
                label_list.add(instance["category"])
        
        # print(label_list)
        label_list = list(label_list)
        
        def getConfig(raw_labels):

            label2id = {}
            id2label = {}

            for i, class_name in enumerate(raw_labels):
                label2id[class_name] = str(i)
                id2label[str(i)] = class_name

            return label2id, id2label

        label2id, id2label = getConfig(label_list)
        
        if cfg.classic==False:
            
                
            thismodel = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=len(label_list))
            model_engine, _, _, _ = cramming.load_backend(thismodel, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
            # Comment if there is no checkpoints
            model_engine.load_checkpoint(cfg_arch, model_file)
            #model_engine.train()
            model = model_engine.model
        
        else:
            tokenizer = AutoTokenizer.from_pretrained("camembert-base")
            model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=len(label_list))
        
        
        model.config.label2id = label2id
        model.config.id2label = id2label
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["text"], truncation=True, max_length=128)
            if cfg.is_signature:
                labels = [int(label2id[label]) for label in examples["signature"]]
            else:
                labels = [int(label2id[label]) for label in examples["category"]]
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
        test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
        
        data_collator = DataCollatorWithPadding(tokenizer)

        print(train_dataset)
        print(test_dataset)
        
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

        #model = AutoModelForSequenceClassification.from_pretrained(checkpoint_name, num_labels=len(label_list))

        output_name = f"BERTinchamps-RTBFCorpus-{task}-{str(uuid.uuid4().hex)}"
        
        args = TrainingArguments(
            output_name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate = 2e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            metric_for_best_model="accuracy", # accuracy/f1?
            load_best_model_at_end=True,
            greater_is_better=True,
        )
        
        trainer = Trainer(
            model,
            args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=test_tokenized_datasets,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        
        trainer.train()
        trainer.evaluate()
        
        predictions, labels, _ = trainer.predict(test_tokenized_datasets)
        predictions = np.argmax(predictions, axis=1)

        true_predictions = [label_list[p] for p in predictions]
        true_labels = [label_list[l] for l in labels]

        f1_score = classification_report(true_labels, true_predictions, digits=4)
        print(f1_score)
if __name__ == "__main__":
    launch()  