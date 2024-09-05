import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from balm.config import BalmConfig, BalmMoEConfig
from balm.data import load_dataset, DataCollator
from balm.models import (
    BalmForMaskedLM,
    BalmModel,
    BalmMoEForMaskedLM,
)
from balm.tokenizer import Tokenizer

from transformers import (
    AutoTokenizer,
    RobertaTokenizer,
    RobertaForMaskedLM,
    pipeline,
)

from itertools import chain
import torch
import torch.nn.functional as F
import abstar
import abutils

#prepping data
#cdr_df1 = pd.read_csv("/home/jovyan/shared/karenna/deberta_help/deberta_test.csv")
cdr_df2 = pd.read_csv("/home/jovyan/shared/karenna/data/lc-coherence-data/with_cdrs/jaffe_sequence_cdr_paired.csv")
all_df = pd.read_csv("/home/jovyan/shared/karenna/data/lc-coherence-data/all.tsv", sep='\t')
test_df = pd.read_csv("/home/jovyan/shared/Sarah/training-data/jaffe_lc-coherence/paired/LC-coherence_90-5-5/test.csv")

test_df = test_df.rename(columns={'roberta_paired': 'sequence'})
cdr_df2['sequence'] = cdr_df2['sequence_aa'].str.replace('<cls><cls>', '</s>')
df = pd.merge(test_df, cdr_df2, on='sequence', how='inner')
df = df.rename(columns={'cdr_mask':'cdr_mask_s'})
#df = pd.merge(df, cdr_df2, on='sequence', how='inner')
#df = df.rename(columns={'cdr_mask':'cdr_mask_cls'})

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

df = pd.merge(df, alldf_pivoted, on='sequence_id', how='inner')
del all_df, alldf_pivoted
df = df.sample(n = 1000, random_state = 0)

# aa sequence tokenizer
tokenizer = Tokenizer(vocab="/home/jovyan/shared/simone/BALM_development/vocab.json")
vocab_list = list(tokenizer.vocab.keys())

#MoE Model
config = BalmMoEConfig(
    expert_choice_router=True,
    embed_dim=960,
    ffn_dim=3840,
    num_experts=16,
    num_heads=20,
    num_shared_experts=1,
    alternate_sparsity=True,
    expert_capacity=128,
    router_z_loss_coef=0.01,
    router_aux_loss_coef=0.01,
    vocab_size=tokenizer.vocab_size,
)
model = BalmMoEForMaskedLM.from_pretrained("/home/jovyan/shared/simone/BALM_development_51324/training_runs/balmMoE_expertchoiceBig_1shared_altern_052924/balmMoE_expertchoiceBig_1shared_altern_052924/model")

model = model.to("cuda")

model2 = RobertaForMaskedLM.from_pretrained("/home/jovyan/shared/Sarah/models/BALM-paired_LC-coherence_90-5-5-split_122222").to("cuda")
tokenizer2 = RobertaTokenizer.from_pretrained("/home/jovyan/shared/Sarah/BALM-paper/pre-training/BALM/tokenizer")

def infer_and_group_stats(model, tokenizer, dnaH, dnaL, cdr):
    losses = []
    predictions = ""
    scores = []
    perplexities = []
    muts = 0

    seqs = abstar.run([abutils.core.sequence.Sequence(dnaH, id='a')]
                      + [abutils.core.sequence.Sequence(dnaL, id='a')], output_type="airr")
    pair = abutils.core.pair.assign_pairs(seqs)

    germ_residues = f'{abutils.tl.translate(pair[0].heavy["germline_alignment"])}'+"<cls><cls>" + f'{abutils.tl.translate(pair[0].light["germline_alignment"])}'
    seq_residues = f'{abutils.tl.translate(pair[0].heavy["sequence_alignment"])}'+"<cls><cls>" + f'{abutils.tl.translate(pair[0].light["sequence_alignment"])}'

    germ_residues = germ_residues.replace("*", ".")
    seq_residues = seq_residues.replace("*", ".")
    
    with torch.no_grad():
        sep = "<cls><cls>"
        sep_idx = seq_residues.find(sep)
        heavy = seq_residues[:sep_idx]
        light = seq_residues[sep_idx + len(sep):]

        unmasked = tokenizer(seq_residues, return_tensors = "pt", padding=True, truncation=True, max_length=320)["input_ids"]
        ranges = [range(sep_idx), range(sep_idx + len(sep), len(seq_residues))]
        total_len = sum(len(i) for i in ranges)

        masked_dict = {}

        # model iteratively predicts each residue
        for i in chain(*ranges):
        #for i in tqdm(chain(*ranges), total=total_len, leave=False):
            masked = seq_residues[:i] + "<mask>" + seq_residues[i+1:]
            masked_dict['text'] = masked
            tokenized = tokenizer(masked_dict['text'], return_tensors = "pt",  padding=True, truncation=True, max_length=320)
            mask_pos = (tokenized["input_ids"][0] == 32).nonzero(as_tuple=True)[0]
            labels = torch.where(tokenized["input_ids"][0] == 32, unmasked[0], torch.tensor(-100))
            output = model(torch.tensor(tokenized["input_ids"][0]).unsqueeze(0).to('cuda'), labels = labels.to('cuda'))

            logits = output['logits'].to('cpu')

            # germline residue
            g = tokenizer.vocab[germ_residues[i]]

            # prediction confidence
            prob = logits[0, mask_pos].softmax(dim=-1)[0, g].numpy()
            scores.append(prob)

            # loss
            loss = output['loss'].to('cpu').numpy()
            losses.append(loss)
            
            # perplexity
            ce_loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1)) # i think this is the same as output.loss.item()
            perplexities.append(float(torch.exp(ce_loss)))

            if seq_residues[i] != germ_residues[i]:
                muts += 1

        # group stats by region
        # find indices splitting regions (fwrs and cdrs in heavy and light chains)
        #cdr_idxs = [0] + [i for i in range(len(cdr)) if cdr[i] != cdr[i-1]] + [len(cdr)]
        #cdr_idxs.insert(7, sep_idx)
        
        
        # prediction confidence
        mean_scores = np.mean(scores)

     
        # loss (mean or median? how best to aggregate since they are different lengths ,,)
        median_loss = np.median(losses)
        #region_mean_loss = [np.mean(losses[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]

        
        # perplexity
        mean_perplexity = np.mean(perplexities)

       # percent mutated
        pct_mut = (muts/len(seq_residues.replace(sep, "")))*100

        return {
            "sequence": seq_residues.replace(sep, ""),
            "heavy": heavy,
            "light": light,
            "mean_germ_score": mean_scores,
            "median_loss": median_loss,
            "mean_perplexity": mean_perplexity,
            "percent_mutated": pct_mut,
            "score": scores,
            "loss": losses,
            "perplexity": perplexities
        }

def infer_and_group_stats2(model, tokenizer, dnaH, dnaL, cdr):
    losses = []
    predictions = ""
    scores = []
    perplexities = []
    muts = 0

    seqs = abstar.run([abutils.core.sequence.Sequence(dnaH, id='a')]
                      + [abutils.core.sequence.Sequence(dnaL, id='a')], output_type="airr")
    pair = abutils.core.pair.assign_pairs(seqs)

    germ_residues = f'{abutils.tl.translate(pair[0].heavy["germline_alignment"])}'+"</s>" + f'{abutils.tl.translate(pair[0].light["germline_alignment"])}'
    seq_residues = f'{abutils.tl.translate(pair[0].heavy["sequence_alignment"])}'+"</s>" + f'{abutils.tl.translate(pair[0].light["sequence_alignment"])}'

    with torch.no_grad():
        sep = "</s>"
        sep_idx = seq_residues.find(sep)
        heavy = seq_residues[:sep_idx]
        light = seq_residues[sep_idx + len(sep):]
        unmasked = tokenizer(seq_residues, return_tensors = "pt").to("cuda")["input_ids"]
        ranges = [range(sep_idx), range(sep_idx + len(sep), len(seq_residues))]
        total_len = sum(len(i) for i in ranges)
        # model iteratively predicts each residue
        for i in chain(*ranges):
        #for i in tqdm(chain(*ranges), total=total_len, leave=False):
            masked = seq_residues[:i] + "<mask>" + seq_residues[i+1:]
            tokenized = tokenizer(masked, return_tensors="pt").to("cuda")
            mask_pos = (tokenized.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            labels = torch.where(tokenized.input_ids == tokenizer.mask_token_id, unmasked, -100)
            output = model(**tokenized, labels = labels.to("cuda"))
            logits = output.logits
            
            # germline residue
            if germ_residues[i] == "*" or germ_residues[i] == "-" or germ_residues[i] == 'X':
                g = tokenizer.get_vocab()["<unk>"]
            else:
                g = tokenizer.get_vocab()[germ_residues[i]]

            # prediction confidence
            prob = logits[0, mask_pos].softmax(dim=-1)[0, g].cpu().numpy()
            scores.append(prob)

            # loss
            loss = output.loss.item()
            losses.append(loss)
            
            # perplexity
            ce_loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1)) # i think this is the same as output.loss.item()
            perplexities.append(float(torch.exp(ce_loss)))

            if seq_residues[i] != germ_residues[i]:
                muts += 1

        # group stats by region
        # find indices splitting regions (fwrs and cdrs in heavy and light chains)
        #cdr_idxs = [0] + [i for i in range(len(cdr)) if cdr[i] != cdr[i-1]] + [len(cdr)]
        #cdr_idxs.insert(7, sep_idx)
        
        
        # prediction confidence
        mean_scores = np.mean(scores)

     
        # loss (mean or median? how best to aggregate since they are different lengths ,,)
        median_loss = np.median(losses)
        #region_mean_loss = [np.mean(losses[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]

        
        # perplexity
        mean_perplexity = np.mean(perplexities)

       # percent mutated
        pct_mut = (muts/len(seq_residues.replace(sep, "")))*100

        return {
            "sequence": seq_residues.replace(sep, ""),
            "heavy": heavy,
            "light": light,
            "mean_germ_score": mean_scores,
            "median_loss": median_loss,
            "mean_perplexity": mean_perplexity,
            "percent_mutated": pct_mut,
            "score": scores,
            "loss": losses,
            "perplexity": perplexities
        }

inference_data = []
sequences = list(df.iterrows())

for _id, row in tqdm(sequences):
    d = infer_and_group_stats2(
        model2, 
        tokenizer2, 
        row['sequence_0'],
        row['sequence_1'], 
        row['cdr_mask_s']
    )
    inference_data.append(d)
        
inference_df = pd.DataFrame(inference_data)

inference_df.to_csv("./inference/jaffe_paired_test_mean_germline_scores_balm.csv")

inference_data = []
sequences = list(df.iterrows())

for _id, row in tqdm(sequences):
    d = infer_and_group_stats(
        model, 
        tokenizer, 
        row['sequence_0'],
        row['sequence_1'], 
        row['cdr_mask_s']
    )
    inference_data.append(d)
        
inference_df = pd.DataFrame(inference_data)

inference_df.to_csv("./inference/jaffe_paired_test_mean_germline_scores_balmMoE.csv")