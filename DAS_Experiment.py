#!/usr/bin/env python
# coding: utf-8

# This code is an adapted version from the notebook:
# https://github.com/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/Boundless_DAS.ipynb

# This notebook was made to reproducee the key results of the Boundless DAS paper https://arxiv.org/pdf/2305.08809 

# We adapted it to work on the tasks given in our research

# The original code was made by: 
__author__ = "Zhengxuan Wu"

# And the adapted version is:
__version__ = "10/05/2023"


# Used Libraries:
import torch
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch import nn
from collections import OrderedDict
import pyvene
from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import set_seed, count_parameters
from accelerate import init_empty_weights, infer_auto_device_map
import transformers 
import LLM_Tasks
import json
from transformers import AutoTokenizer,AutoModelForCausalLM,LlamaConfig,LlamaForCausalLM, LlamaTokenizer,PreTrainedTokenizerFast


# Helper functions:
def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


#Model Settings:
name="meta-llama/Llama-3.2-3B"
device_input='cuda:1' #has to be the device of the input layer as well as the one which we change.
device_map=OrderedDict([('model.embed_tokens', 1), 
                        ('model.layers.0', 1), 
                        ('model.layers.1', 1), 
                        ('model.layers.2', 1), 
                        ('model.layers.3', 1), 
                        ('model.layers.4', 1), 
                        ('model.layers.5', 1), 
                        ('model.layers.6', 0), 
                        ('model.layers.7', 0), 
                        ('model.layers.8', 0), 
                        ('model.layers.9', 0), 
                        ('model.layers.10', 0), 
                        ('model.layers.11', 0), 
                        ('model.layers.12', 0), 
                        ('model.layers.13', 0), 
                        ('model.layers.14', 0), 
                        ('model.layers.15', 1), 
                        ('model.layers.16', 1), 
                        ('model.layers.17', 1), 
                        ('model.layers.18', 0), 
                        ('model.layers.19', 0), 
                        ('model.layers.20', 0), 
                        ('model.layers.21', 0), 
                        ('model.layers.22', 0), 
                        ('model.layers.23', 0), 
                        ('model.layers.24', 0), 
                        ('model.layers.25', 0), 
                        ('model.layers.26', 0), 
                        ('model.layers.27', 0), 
                        ('model.norm', 1), 
                        ('model.rotary_emb', 1), 
                        ('lm_head', 1)])

tokenizer = AutoTokenizer.from_pretrained(name)

# Experiment settings
change_layer=16
batchsize=8
Max_examples_token_length=600
num_eval_set=1000
num_train_set=10000
Changed_Dimension=0 #in 2D setting can be 0 or 1


# Task Preparation
Use_Context=True
num_range = list(range(23))
Task_1=LLM_Tasks.Regression_Task_Int(
    tokenizer,
    Display_Context=Use_Context,
    max_examples_token_length=Max_examples_token_length
)

Task_1.New_Task()
prompt_len=len(tokenizer(Task_1.Generate_Task()[0]).input_ids)
which_to_swap=prompt_len-1


#Preparation of model:
dtype=torch.bfloat16
config = LlamaConfig.from_pretrained(name)
llama = LlamaForCausalLM.from_pretrained(
    name,
    config=config,
    device_map=device_map,
    torch_dtype=dtype,  # save memory
)


_ = llama.eval()  # no gradients



# Prepare prealign dataset
raw_prealign =  Task_1.get_Eval_Dataset(num_eval_set)
prealign_dataset = Dataset.from_dict(
    {"input_ids": raw_prealign[0], "labels": raw_prealign[1]}
)
prealign_dataset.set_format("torch", columns=["input_ids", "labels"])
prealign_dataloader = DataLoader(prealign_dataset, batch_size=8)




#Evaluation
total_count = 0
correct_count = 0
squared_diff = 0
total_int_count = 0
with torch.no_grad():
    for step, inputs in enumerate(tqdm(prealign_dataloader)):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(llama.device)

        # aligning forward!
        outputs = llama(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )

        actual_test_labels = inputs["labels"][:, -1]
        pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)
        correct_labels = actual_test_labels == pred_test_labels
        for ii in range(len(inputs["input_ids"])):
            pred_int=tokenizer.decode([pred_test_labels[ii]])
            true_int=tokenizer.decode([actual_test_labels[ii]])
            if is_integer(pred_int):
                squared_diff+=(int(pred_int)-int(true_int))**2
                total_int_count+=1
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
current_acc = round(correct_count / total_count, 2)
current_diff = round((squared_diff / total_int_count), 2)
current_notint = round((total_count-total_int_count) / num_eval_set, 2)
print("Needs to be good:")
print(f"Prealign task accuracy: {current_acc}")
print(f"Squared difference: {current_diff}")
print(f"Non integers predicted: {current_notint}")


# Prepare raining dataset for trainable intervention:
set_seed(42)

raw_data =  Task_1.get_Dataset_DAS(num_train_set,Changed_Dimension)
raw_train = (
    raw_data[0][:int(num_train_set/10*8)],
    raw_data[1][:int(num_train_set/10*8)],
    raw_data[2][:int(num_train_set/10*8)],
    raw_data[3][:int(num_train_set/10*8)],
)
raw_eval = (
    raw_data[0][int(num_train_set/10*8):int(num_train_set/10*9)],
    raw_data[1][int(num_train_set/10*8):int(num_train_set/10*9)],
    raw_data[2][int(num_train_set/10*8):int(num_train_set/10*9)],
    raw_data[3][int(num_train_set/10*8):int(num_train_set/10*9)],
)
raw_test = (
    raw_data[0][int(num_train_set/10*9):],
    raw_data[1][int(num_train_set/10*9):],
    raw_data[2][int(num_train_set/10*9):],
    raw_data[3][int(num_train_set/10*9):],
)
train_dataset = Dataset.from_dict(
    {
        "input_ids": raw_train[0],
        "source_input_ids": raw_train[1],
        "labels": raw_train[2],
        "intervention_ids": raw_train[3], 
    }
).with_format("torch")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batchsize,
)
eval_dataset = Dataset.from_dict(
    {
        "input_ids": raw_eval[0],
        "source_input_ids": raw_eval[1],
        "labels": raw_eval[2],
        "intervention_ids": raw_eval[3], 
    }
).with_format("torch")
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=batchsize,
)
test_dataset = Dataset.from_dict(
    {
        "input_ids": raw_test[0],
        "source_input_ids": raw_test[1],
        "labels": raw_test[2],
        "intervention_ids": raw_test[3], 
    }
).with_format("torch")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batchsize,
)


# Boundless DAS on Position-aligned Tokens preparations:
def simple_boundless_das_position_config(model_type, intervention_type, layer):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,              
                intervention_type,  
            ),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return config
    
#Possible outputs to analyse:
#['block_input', 'block_output', 'mlp_activation', 'mlp_output', 'mlp_input', 'attention_value_output', 'head_attention_value_output', 'attention_output', 'attention_input', 'query_output', 'key_output', 'value_output', 'head_query_output', 'head_key_output', 'head_value_output']
config = simple_boundless_das_position_config(
    type(llama), "block_output", str(change_layer)
)
intervenable = IntervenableModel(config, llama)
intervenable.disable_model_gradients()


for k,v in intervenable.interventions.items():
    v[0].intervention_population=nn.Parameter(v[0].intervention_population.to(device_input), requires_grad=False)
    _=v[0].rotate_layer.to(device_input)
    v[0].intervention_boundaries=nn.Parameter(v[0].intervention_boundaries.to(device_input))


t_total = int(len(train_dataloader) * 3)
warm_up_steps = 0.1 * t_total
optimizer_params = []
for k, v in intervenable.interventions.items():
    optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
    optimizer_params += [{"params": v[0].intervention_boundaries, "lr": 1e-2}]
optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
)

def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    squared_diff=0
    total_int_count=0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -1].cpu()
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1).cpu()
        #print(actual_test_labels, pred_test_labels)
        correct_labels = actual_test_labels == pred_test_labels
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
        for ii in range(len(inputs["input_ids"])):
            pred_int=tokenizer.decode([pred_test_labels[ii]])
            true_int=tokenizer.decode([actual_test_labels[ii]])
            try:
                squared_diff+=(int(pred_int)-int(true_int))**2
                total_int_count+=1
            except:
                pass
    accuracy = round(correct_count / total_count, 2)
    current_diff = round((squared_diff / total_int_count), 2)
    current_notint = round((total_count-total_int_count) / num_eval_set, 2)
    return {"accuracy": accuracy,"squared_diff": current_diff,"not int": current_notint}


# Boundless DAS on Position-aligned Tokens training:
epochs = 3
gradient_accumulation_steps = 4
total_step = 0
target_total_step = len(train_dataloader) * epochs
temperature_start = 50.0
temperature_end = 0.1
temperature_schedule = (
    torch.linspace(temperature_start, temperature_end, target_total_step)
    .to(torch.bfloat16)
    .to(device_input)
)
intervenable.set_temperature(temperature_schedule[total_step])


def calculate_loss(logits, labels):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, intervenable.model_config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    for k, v in intervenable.interventions.items():
        boundary_loss = 1.0 * v[0].intervention_boundaries.sum()
    loss += boundary_loss

    return loss


intervenable.model.train()  # train enables drop-off but no grads
print("llama trainable parameters: ", count_parameters(intervenable.model))
print("intervention trainable parameters: ", intervenable.count_parameters())
train_iterator = trange(0, int(epochs), desc="Epoch")
for epoch in train_iterator:
    epoch_iterator = tqdm(
        train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
    )
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device_input)
                #print(inputs[k].get_device())
        b_s = inputs["input_ids"].shape[0]
        #print(inputs["input_ids"])
        #print("*"*100)
        #print(inputs["source_input_ids"])
        _, counterfactual_outputs = intervenable(
            {"input_ids": inputs["input_ids"]},
            [{"input_ids": inputs["source_input_ids"]}],
            {"sources->base": which_to_swap},  # swap 80th token
        )
        eval_metrics = compute_metrics(
            [counterfactual_outputs.logits], [inputs["labels"]]
        )

        # loss and backprop
        loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"])
        loss_str = round(loss.item(), 2)
        epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if total_step % gradient_accumulation_steps == 0:
            if not (gradient_accumulation_steps > 1 and total_step == 0):
                optimizer.step()
                scheduler.step()
                intervenable.set_zero_grad()
                intervenable.set_temperature(temperature_schedule[total_step])
        total_step += 1


# Boundless DAS on Position-aligned Tokens evaluations:
eval_labels = []
eval_preds = []
with torch.no_grad():
    epoch_iterator = tqdm(test_dataloader, desc=f"Test")
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device_input)
        b_s = inputs["input_ids"].shape[0]
        _, counterfactual_outputs = intervenable(
            {"input_ids": inputs["input_ids"]},
            [{"input_ids": inputs["source_input_ids"]}],
            {"sources->base": which_to_swap},  # swap 80th token
        )
        eval_labels += [inputs["labels"]]
        eval_preds += [counterfactual_outputs.logits.cpu()]
eval_metrics = compute_metrics(eval_preds, eval_labels)
print(eval_metrics)



