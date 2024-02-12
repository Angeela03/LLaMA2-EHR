import os
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM, get_peft_config, \
    get_peft_model, TaskType
from datasets import load_dataset, load_metric
import bitsandbytes as bnb
from functools import partial
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import sys
import re
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
import numpy as np

code = "oud"

if code == "oud":
    code_text = "Opioid Use Disorder"
elif code == "sud":
    code_text = "Substance Use Disorder"
elif code == "diabetes":
    code_text = "Diabetes"
else:
    print("Error in the code")
    sys.exit(1)

text_column = "Text"
label_column = "Text_label"
prompt = "prompt2"
output = "MIMIC_new_task_finetune_"
max_length = 4096
model_id = '/llama2/Llama-2-7b-chat-hf'
data_path = '/llama2/data/'
save_name = output + code + "_" + prompt + '_yes_no'

output_dir = os.path.join("outputs", save_name)
log_dir = os.path.join("logs", save_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    warmup_steps=50,
    learning_rate=5e-4,
    save_strategy='steps',
    evaluation_strategy='steps',
    # load_best_model_at_end=True,
    # metric_for_best_model='eval_loss',
    save_total_limit=20,
    fp16=True,
    output_dir=output_dir,
    optim="paged_adamw_8bit",
    eval_steps=2900,
    logging_steps=580,
    save_steps=2900,
    # eval_steps=250,
    # logging_steps=50,
    # save_steps=250,
    logging_dir=log_dir,
)

exact_match = load_metric("exact_match")
tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


def create_bnb_config():
    """Quantization config"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=8,  # dimension of the updated matrices
        lora_alpha=32,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    return config


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def create_prompt(examples):
    # Initialize static strings for the prompt template
    INTRO_BLURB = "Given a patient's past medical history, predict whether the patient will have a future diagnosis of " + code_text + ". Return 'Yes' or 'No' after the XML tag <Diagnosis>."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
    instruction = f"{INSTRUCTION_KEY}\n{INTRO_BLURB}"
    input_context = f"{INPUT_KEY}\n{examples[text_column]}" if examples[text_column] else None

    high_low_label = examples[label_column]
    if high_low_label == "High":
        t_label = "Yes"
    elif high_low_label == "Low":
        t_label = "No"
    else:
        print("There is some error with the label")

    response = f"{RESPONSE_KEY}\n<Diagnosis>{t_label}</Diagnosis>"
    end = f"{END_KEY}"

    # Create a list of prompt template elements
    parts = [part for part in [instruction, input_context, response, end] if part]

    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)

    # Store the formatted prompt template in a new key "text"
    examples["prompt"] = formatted_prompt

    return examples


def preprocess_batch(batch, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["prompt"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_dataset(max_length: int, seed, dataset):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset (str): Instruction dataset
    """

    print("Preprocessing dataset...", flush=True)
    dataset = dataset.map(create_prompt)

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['PatientId', 'Text', 'Label', 'Text_label', 'prompt'],
    )

    # Filter out samples that have "input_ids" exceeding "max_length"
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def compute_metrics(eval_preds):
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    labels[labels == -100] = tokenizer.pad_token_id
    labels_decode = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels_yes_no = [re.findall("### Response:\n<Diagnosis>(.*?)</Diagnosis>", i) for i in labels_decode]
    labels_1_0 = [1 if i[0] == "Yes" else 0 for i in labels_yes_no]
    auc_roc = roc_auc_score(labels_1_0, preds)
    pre_auc = average_precision_score(labels_1_0, preds)
    return {"ROC AUC score": round(auc_roc, 2), "PR AUC score": round(pre_auc, 2)}


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    pred_ids_unsqueeze = torch.unsqueeze(pred_ids, -1)

    pred = []

    for i, preds in enumerate(pred_ids_unsqueeze):
        preds_decode = tokenizer.batch_decode(preds, skip_special_tokens=False)
        index = 0
        for j in range(len(preds_decode) - 8):
            if preds_decode[j:j + 8] == ['Response', ':', '\n', '<', 'Di', 'agn', 'osis', '>']:
                index = j + 8
                break
        probs_index = logits[i, index, :]
        next_token_probs = torch.softmax(probs_index, dim=0)
        topk_next_tokens = torch.topk(next_token_probs, 20)

        low_tokens = ["No", "No", "N", "no", "NO"]
        high_tokens = ["Yes", "Yes", "yes", "yes", "YES", "Y"]
        top_k_probs = [(tokenizer.decode(idx), prob) for idx, prob in
                       zip(topk_next_tokens.indices, topk_next_tokens.values)]
        low_sum = 0
        high_sum = 0

        for k, v in top_k_probs:
            if k in low_tokens:
                low_sum += v.item()
            elif k in high_tokens:
                high_sum += v.item()
        arr = [high_sum, low_sum]
        low_high_probs = np.exp(arr) / np.sum(np.exp(arr), axis=0)
        pred.append(low_high_probs[0])
    return torch.as_tensor(pred)


def train(model, preprocessed_dataset):
    train_dataset = preprocessed_dataset["train"]
    eval_dataset = preprocessed_dataset["valid"]

    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    # print_trainable_parameters(model)
    model.print_trainable_parameters()

    # for p in model.parameters():
    #     p.requires_grad = False

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    do_train = True
    # do_eval = True

    # Launch training
    print("Training...", flush=True)

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics, flush=True)

        # Saving model
    print("Saving last checkpoint of the model...", flush=True)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


def main():
    data_files = {"train": output + "train_" + code + "_" + prompt + ".csv",
                  "valid": output + "valid_" + code + "_" + prompt + ".csv"}
    dataset = load_dataset(data_path, data_files=data_files)

    bnb_config = create_bnb_config()
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available ressources
    )
    preprocessed_dataset = preprocess_dataset(max_length, 7, dataset)
    train(model, preprocessed_dataset)


if __name__ == main():
    main()
