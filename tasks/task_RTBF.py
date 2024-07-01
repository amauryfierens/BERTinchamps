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

@hydra.main(config_path="cramming/config", config_name="cfg_rtbf", version_base="1.1")
def launch(cfg):
    task = "seq_classif"
    batch_size = 32
    EPOCHS = 20
    
    # Directory containing your JSON file
    pre_path = ""
    dir_path = 'local_data/Testing_corpora/'
    k = cfg.nbr
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
        local_checkpoint_folder =  os.path.join(pre_path, cfg.base_dir, cfg.name,"checkpoints")
        
        all_checkpoints = [f for f in os.listdir(local_checkpoint_folder)]
        checkpoint_paths = [os.path.join(local_checkpoint_folder, f) for f in all_checkpoints]
        checkpoint_name = max(checkpoint_paths, key=os.path.getmtime)
        
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(checkpoint_name, "tokenizer.json"))
        tokenizer.pad_token = "<pad>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.unk_token = "<unk>"
        tokenizer.sep_token = "<sep>"
        tokenizer.cls_token = "<cls>"
        tokenizer.mask_token = "<mask>"
        with open(os.path.join(checkpoint_name, "model_config.json"), "r") as f:
            cfg_arch = OmegaConf.create(json.load(f))
        if cfg.eval.arch_modifications is not None:
            cfg_arch = OmegaConf.merge(cfg_arch, cfg.eval.arch_modifications)
        model_file = os.path.join(checkpoint_name, "model.pth")
        
        setup = cramming.utils.system_startup(cfg)
        
        thismodel = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=len(label_list))
        model_engine, _, _, _ = cramming.load_backend(thismodel, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
        
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

    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)
    
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