import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm
tqdm.pandas(leave = False)

from transformers import (
    AutoTokenizer,
    RobertaTokenizer,
    RobertaForMaskedLM,
    pipeline,
)

from balm.config import BalmConfig, BalmMoEConfig
from balm.data import load_dataset, DataCollator
from balm.models import (
    BalmForMaskedLM,
    BalmModel,
    BalmMoEForMaskedLM,
)
from balm.tokenizer import Tokenizer

from itertools import chain
import torch
import torch.nn.functional as F
import abstar
import abutils

#prepping data
cdr_df1 = pd.read_csv("./inference/regionalinference_test.csv")
cdr_df2 = pd.read_csv("./inference/jaffe_sequence_cdr_paired.csv")
all_df = pd.read_csv("./inference/all.tsv", sep='\t')
test_df = pd.read_csv("./inference/test.csv")

test_df = test_df.rename(columns={'roberta_paired': 'sequence'})
cdr_df2['sequence'] = cdr_df2['sequence_aa'].str.replace('<cls><cls>', '</s>')
df = pd.merge(test_df, cdr_df1, on='sequence', how='inner')
df = df.rename(columns={'cdr_mask':'cdr_mask_s'})
df = pd.merge(df, cdr_df2, on='sequence', how='inner')
df = df.rename(columns={'cdr_mask':'cdr_mask_cls'})

df['sequence_id'] = df['name'] +'-' + df['dataset'].astype(str)
all_df = all_df[all_df['sequence_id'].isin(df['sequence_id'])]

# Create an auxiliary column to distinguish the two rows in each group
all_df['row_num'] = all_df.groupby('sequence_id').cumcount()

# Pivot the table to get the desired stacked format
alldf_pivoted = all_df.pivot(index='sequence_id', columns='row_num')

# Flatten the MultiIndex columns
alldf_pivoted.columns = [f'{col}_{row_num}' for col, row_num in alldf_pivoted.columns]

# Reset the index to turn it back into a column
alldf_pivoted.reset_index(inplace=True)

#device
df = pd.merge(df, alldf_pivoted, on='sequence_id', how='inner')
del all_df, alldf_pivoted

df = df.sample(n = 1000, random_state = 0)
df = df[(df['v_mutation_count_aa_0'] == 0) | (df['v_mutation_count_aa_1'] == 0)]

# aa sequence tokenizer
tokenizer = Tokenizer(vocab="./balm/vocab.json")
vocab_list = list(tokenizer.vocab.keys())
model = BalmMoEForMaskedLM.from_pretrained("./training_runs/balmMoE_expertchoice_1shared_altern_052924/model")
model = model.to("cuda")


model2 = RobertaForMaskedLM.from_pretrained("/models/BALM-paired_LC-coherence_90-5-5-split_122222").to("cuda")
tokenizer2 = RobertaTokenizer.from_pretrained("/BALM-paper/pre-training/BALM/tokenizer")

def infer_and_group_stats2(model, tokenizer, seq, cdr):
    losses = []
    predictions = ""
    scores = []
    perplexities = []
    with torch.no_grad():
        sep = "</s>"
        sep_idx = seq.find(sep)
        heavy = seq[:sep_idx]
        light = seq[sep_idx + len(sep):]
        unmasked = tokenizer(seq, return_tensors = "pt").to("cuda")["input_ids"]
        ranges = [range(sep_idx), range(sep_idx + len(sep), len(seq))]
        total_len = sum(len(i) for i in ranges)
        # model iteratively predicts each residue
        for i in chain(*ranges):
        #for i in tqdm(chain(*ranges), total=total_len, leave=False):
            masked = seq[:i] + "<mask>" + seq[i+1:]
            tokenized = tokenizer(masked, return_tensors="pt").to("cuda")
            mask_pos = (tokenized.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            labels = torch.where(tokenized.input_ids == tokenizer.mask_token_id, unmasked, -100)
            output = model(**tokenized, labels = labels.to("cuda"))
            logits = output.logits
            # predicted aa
            pred_token = logits[0, mask_pos].argmax(axis=-1)
            predictions+=tokenizer.decode(pred_token)
            # prediction confidence
            prob = logits[0, mask_pos].softmax(dim=-1).topk(1)[0].item()
            scores.append(prob)
            # loss
            loss = output.loss.item()
            losses.append(loss)
            # perplexity
            ce_loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1)) # i think this is the same as output.loss.item()
            perplexities.append(float(torch.exp(ce_loss)))
        # group stats by region
        # find indices splitting regions (fwrs and cdrs in heavy and light chains)
        cdr_idxs = [0] + [i for i in range(len(cdr)) if cdr[i] != cdr[i-1]] + [len(cdr)]
        cdr_idxs.insert(7, sep_idx)
        # accuracy
        predictions_by_region = [predictions[cdr_idxs[n]:cdr_idxs[n+1]] for n in range(len(cdr_idxs)-1)]
        seq_by_region = [seq.replace(sep, "")[cdr_idxs[n]:cdr_idxs[n+1]] for n in range(len(cdr_idxs)-1)]
        region_mean_acc = [sum(true[i] == predict[i] for i in range(len(true)))/len(true) for true, predict in zip(seq_by_region, predictions_by_region)]
        # prediction confidence
        region_mean_scores = [np.mean(scores[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]
        # loss (mean or median? how best to aggregate since they are different lengths ,,)
        region_median_loss = [np.median(losses[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]
        # perplexity
        region_mean_perplexity = [np.mean(perplexities[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]
        return {
            "sequence": seq.replace(sep, ""),
            "heavy": heavy,
            "light": light,
            "cdr_indices": cdr_idxs,
            "prediction": heavy + light,  #heavy + sep + light
            "accuracy_by_region": region_mean_acc,
            "score_by_region": region_mean_scores,
            "loss_by_region": region_median_loss,
            "perplexity_by_region": region_mean_perplexity,
            "score": scores,
            "loss": losses,
            "perplexity": perplexities
        }

def infer_and_group_stats(model, tokenizer, seq, cdr):
    losses = []
    predictions = ""
    scores = []
    perplexities = []
    with torch.no_grad():
        sep = "<cls><cls>"
        sep_idx = seq.find(sep)
        heavy = seq[:sep_idx]
        light = seq[sep_idx + len(sep):]
        unmasked = tokenizer(seq, return_tensors = "pt", padding=True, truncation=True, max_length=320)["input_ids"]
        ranges = [range(sep_idx), range(sep_idx + len(sep), len(seq))]
        total_len = sum(len(i) for i in ranges)
        masked_dict = {}
        # model iteratively predicts each residue
        for i in chain(*ranges):
            masked = seq[:i] + "<mask>" + seq[i+1:]
            masked_dict['text'] = masked
            tokenized = tokenizer(masked_dict['text'], return_tensors = "pt",  padding=True, truncation=True, max_length=320)
            mask_pos = (tokenized["input_ids"][0] == 32).nonzero(as_tuple=True)[0]
            labels = torch.where(tokenized["input_ids"][0] == 32, unmasked[0], torch.tensor(-100))
            output = model(torch.tensor(tokenized["input_ids"][0]).unsqueeze(0).to('cuda'), labels = labels.to('cuda'))
            logits = output['logits'].to('cuda')
                #prediction
            pred_token = torch.argmax(logits[0][mask_pos])
            if vocab_list[pred_token] == '<cls>':
                predictions+='c'
            elif vocab_list[pred_token] == '<pad>':
                predictions+='p'
            elif vocab_list[pred_token] == '<eos>':
                predictions+='e'
            elif vocab_list[pred_token] == '<unk>':
                predictions+='u'
            elif vocab_list[pred_token] == 'null_1' or vocab_list[pred_token] == '<mask>':
                predictions+='m'
            else:
                predictions+=vocab_list[pred_token]
            # prediction confidence
            prob = logits[0, mask_pos].softmax(dim=-1).topk(1)[0].item()
            #print(prob)
            scores.append(prob)
            # loss
            loss = output['loss'].cpu()
            losses.append(loss)
            # perplexity
            ce_loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size).to('cuda'), labels.view(-1).to('cuda')) # i think this is the same as output.loss.item()
            perplexities.append(float(torch.exp(ce_loss)))
        # group stats by region
        # find indices splitting regions (fwrs and cdrs in heavy and light chains)
        cdr_idxs = [0] + [i for i in range(len(cdr)) if cdr[i] != cdr[i-1]] + [len(cdr)]
        cdr_idxs.insert(7, sep_idx)
        # accuracy
        seq = seq.replace("<cls>", "")
        if len(seq) != len(predictions):
            print("problem")
            print(len(seq))
            print(len(predictions))
        predictions_by_region = [predictions[cdr_idxs[n]:cdr_idxs[n+1]] for n in range(len(cdr_idxs)-1)]
        seq_by_region = [seq.replace(sep, "")[cdr_idxs[n]:cdr_idxs[n+1]] for n in range(len(cdr_idxs)-1)]
        region_mean_acc = [sum(true[i] == predict[i] for i in range(len(true)))/len(true) for true, predict in zip(seq_by_region, predictions_by_region)]
        # prediction confidence
        region_mean_scores = [np.mean(scores[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]
        # loss (mean or median? how best to aggregate since they are different lengths ,,)
        region_median_loss = [np.median(losses[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]
        #region_mean_loss = [np.mean(losses[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]
        # perplexity
        region_mean_perplexity = [np.mean(perplexities[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]
        return {
            "sequence": seq.replace(sep, ""),
            "heavy": heavy,
            "light": light,
            "cdr_indices": cdr_idxs,
            "prediction": heavy + light,  #heavy + sep + light
            "accuracy_by_region": region_mean_acc,
            "score_by_region": region_mean_scores,
            "loss_by_region": region_median_loss,
            "perplexity_by_region": region_mean_perplexity,
            "score": scores,
            "loss": losses,
            "perplexity": perplexities
        }

    


inference_data = []
sequences = list(df.iterrows())

for _id, row in tqdm(sequences):
    d = infer_and_group_stats(
        model, 
        tokenizer, 
        row['sequence_aa'], 
        row['cdr_mask_cls']
    )
    inference_data.append(d)
        
inference_df = pd.DataFrame(inference_data)


inference_data = []
sequences = list(df.iterrows())

for _id, row in tqdm(sequences):
    d = infer_and_group_stats2(
        model2, 
        tokenizer2, 
        row['sequence'],
        row['cdr_mask_s']
    )
    inference_data.append(d)
        
inference_df2 = pd.DataFrame(inference_data)

model_stats = {
    "balmMoE":inference_df,
    "balm":inference_df2,
}

stats_list = []

for ratio, df in model_stats.items():
    model_df = df #pd.read_json(data_path)
    model_df = model_df[model_df["cdr_indices"].map(len) == 15] # remove rows without cdr2, would change regional split of stats and not sure how to fix without redoing alignment
    acc_df = pd.DataFrame(list(model_df["accuracy_by_region"]))
    score_df = pd.DataFrame(list(model_df["score_by_region"]))
    loss_df = pd.DataFrame(list(model_df["loss_by_region"]))
    perplexity_df = pd.DataFrame(list(model_df["perplexity_by_region"]))
    
    # the way each metrics' df is made, they should refer to the same instance at the same indices, it doesn't really matter though
    for n in acc_df.columns:
        for i in range(len(acc_df)):
            stats_list.append({
                "region": n,
                "model": ratio,
                "accuracy": acc_df.iloc[i, n],
                "score": score_df.iloc[i, n],
                "loss": loss_df.iloc[i, n],
                "perplexity": perplexity_df.iloc[i, n],
            })

stats_df = pd.DataFrame(stats_list)
stats_df.replace({"region": {0:"HC-FR1", 1:"HC-CDR1", 2:"HC-FR2", 3:"HC-CDR2", 4:"HC-FR3", 5:"HC-CDR3", 6:"HC-FR4", 7:"LC-FR1",
                             8:"LC-CDR1", 9:"LC-FR2", 10:"LC-CDR2", 11:"LC-FR3", 12:"LC-CDR3", 13:"LC-FR4"}}, inplace=True)
stats_df.to_csv("./inference/inferencebyregion_germ_balmMoE_balm.csv")

cdr_len_list = []

for model, data_path in model_stats.items():
    model_df = data_path
    model_df = model_df[model_df["cdr_indices"].map(len) == 15] # remove rows without cdr2, would change regional split of stats and not sure how to fix without redoing alignment
    
    for _id, row in list(model_df.iterrows()):
        stats = [{"region": "hcdr3",
                  "model": model,
                  "length": row["cdr_indices"][6] - row["cdr_indices"][5],
                  "accuracy": row["accuracy_by_region"][5],
                  "score": row["score_by_region"][5],
                  "loss": row["loss_by_region"][5],
                  "perplexity": row["perplexity_by_region"][5],
                 },
                 {"region": "lcdr3",
                  "model": model,
                  "length": row["cdr_indices"][13] - row["cdr_indices"][12],
                  "accuracy": row["accuracy_by_region"][12],
                  "score": row["score_by_region"][12],
                  "loss": row["loss_by_region"][12],
                  "perplexity": row["perplexity_by_region"][12],
                 }]
        cdr_len_list.extend(stats)
        
cdr3_len_df = pd.DataFrame(cdr_len_list)

cdr3_len_df.to_csv("./inference/inferencebyCDRlength_germ_balmMoE_balm.csv")