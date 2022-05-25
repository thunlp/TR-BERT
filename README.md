# TR-BERT

Source code and dataset for "[TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference](https://arxiv.org/abs/2105.11618)".

![model](https://github.com/thunlp/TR-BERT/blob/master/model.jpg)

The code is based on huggaface's [transformers](https://github.com/huggingface/transformers). Thanks to them! 

## Requirement
Install dependencies and [apex](https://github.com/NVIDIA/apex):
```
pip3 install -r requirement.txt
pip3 install --editable transformers
```

## Pretrained models

Download the DistilBERT-3layer and BERT-1024 from [Google Drive](https://drive.google.com/drive/folders/1NzJ5_LVlQ1IORj48zTpYXE-4FGCATsf-?usp=sharing)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/869133d3fdc2400baf30).
## Classfication


Download the IMDB, Yelp, 20News datasets from [Google Drive](https://drive.google.com/drive/folders/1NzJ5_LVlQ1IORj48zTpYXE-4FGCATsf-?usp=sharing)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/869133d3fdc2400baf30).


Download the [Hyperpartisan](https://zenodo.org/record/1489920#.X2DxuWgzaUk) dataset, and randomly split it into train/dev/test set: `python3 split_hyperpartisan.py`


### Train BERT/DistilBERT Model
Use flag `--do train`:
```
python3 run_classification.py  --task_name imdb  --model_type bert  --model_name_or_path bert-base-uncased --data_dir imdb --max_seq_length 512  --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 4 --learning_rate 3e-5 --save_steps 2000  --num_train_epochs 5  --output_dir imdb_models/bert_base  --do_lower_case  --do_eval  --evaluate_during_training  --do_train
```

where *task_name* can be set as *imdb/yelp_f/20news/hyperpartisan* for different tasks and *model type* can be set as *bert/distilbert* for different models.

### Compute Graident for Residual Strategy 
Use flag `--do_eval_grad`.
```
python3 run_classification.py  --task_name imdb  --model_type bert  --model_name_or_path imdb_models/bert_base --data_dir imdb --max_seq_length 512  --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 8  --output_dir imdb_models/bert_base  --do_lower_case  --do_eval_grad
```
This step doesn't supoort data DataParallel or DistributedDataParallel currently and should be done in a single GPU.

### Train the policy network solely
Start from the checkpoint from the task-specific fine-tuned model. Change *model_type* from *bert* to *autobert*, and run with flag `--do_train --train_rl`:
```
python3 run_classification.py  --task_name imdb  --model_type autobert  --model_name_or_path imdb_models/bert_base --data_dir imdb --max_seq_length 512  --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 8 --gradient_accumulation_steps 4 --learning_rate 3e-5 --save_steps 2000  --num_train_epochs 3  --output_dir imdb_models/auto_1  --do_lower_case  --do_train --train_rl --alpha 1 --guide_rate 0.5
```
where *alpha* is the harmonic coefficient for the length punishment and *guide_rate* is the proportion of imitation learning steps. *model_type* can be set as *autobert/distilautobert* for applying token reduction to BERT/DistilBERT.

### Compute Logits for Knowledge Distilation
Use flag `--do_eval_logits`.
```
python3 run_classification.py  --task_name imdb  --model_type bert  --model_name_or_path imdb_models/bert_base --data_dir imdb --max_seq_length 512  --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 8  --output_dir imdb_models/bert_base  --do_lower_case  --do_eval_logits
```
This step doesn't supoort data DataParallel or DistributedDataParallel currently and should be done in a single GPU.


### Train the whole network with both the task-specifc objective and RL objective
Start from the checkpoint from `--train_rl` model and run with flag `--do_train --train_both --train_teacher`:
```
python3 run_classification.py  --task_name imdb  --model_type autobert  --model_name_or_path imdb_models/auto_1 --data_dir imdb --max_seq_length 512  --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 1 --gradient_accumulation_steps 4 --learning_rate 3e-5 --save_steps 2000  --num_train_epochs 3  --output_dir imdb_models/auto_1_both  --do_lower_case  --do_train --train_both --train_teacher --alpha 1
```

### Evaluate
Use flag `--do_eval`:
```
python3 run_classification.py  --task_name imdb  --model_type autobert  --model_name_or_path imdb_models/auto_1_both  --data_dir imdb --max_seq_length 512  --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 1  --output_dir imdb_models/auto_1_both  --do_lower_case  --do_eval --eval_all_checkpoints
```

When the batch size is more than 1 in evaluating, we will remain the same number of tokens for each instance in the same batch.


### Initialize
For IMDB dataset, we find that when we directly initialize the selector with heuristic objective before train 
the policy network solely, we can get a bit better performance. For other datasets, this step makes little change. Run this step with flag `--do_train --train_init`:

```
python3 trans_imdb_rank.py
python3 run_classification.py  --task_name imdb  --model_type initbert  --model_name_or_path imdb_models/bert_base --data_dir imdb --max_seq_length 512  --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 8 --gradient_accumulation_steps 4 --learning_rate 3e-5 --save_steps 2000  --num_train_epochs 3  --output_dir imdb_models/bert_init  --do_lower_case  --do_train --train_init 
```


## Question Answering

Download the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer) dataset.

Download the MRQA dataset with our split] from [Google Drive](https://drive.google.com/drive/folders/1NzJ5_LVlQ1IORj48zTpYXE-4FGCATsf-?usp=sharing)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/869133d3fdc2400baf30).

Download the [HotpotQA](https://github.com/microsoft/Transformer-XH) dataset from the Transformer-XH repository, where paragraphs are retrieved for each question according to TF-IDF, entity linking and hyperlink and re-ranked by BERT re-ranker.

Download the [TriviaQA](https://github.com/mandarjoshi90/linked_qa_datasets) dataset, where paragraphs are re-rank by the linear passage re-ranker in DocQA.

Download the [WikiHop](https://qangaroo.cs.ucl.ac.uk) dataset.

The whole training progress of question answer models is similiar to text classfication models, with flags `--do_train`, `--do_train --train_rl`, `--do_train --train_both --train_teacher` in turn. The codes of each dataset:

SQuAD: `run_squad.py` with flag `version_2_with_negative`

NewsQA / NaturalQA: `run_mrqa.py`

RACE: `run_race_classify.py`

HotpotQA: `run_hotpotqa.py`

TriviaQA: `run_triviaqa.py`

WikiHop: `run_wikihop.py`


## Harmonic Coefficient Lambda

The example harmonic coefficients are shown as follows:

|Dataset | train_rl | train_both |  
|:-: |:-:|:-:  |
|SQuAD 2.0	| 5	|  5  |
|NewsQA	| 3	| 5 |
|NaturalQA | 2	| 2 |
|RACE |0.5 | 0.1 |
|YELP.F	|2 |0.5|
|20News	|1 |1|
|IMDB |1 |1|
|HotpotQA |0.1 |4|
|TriviaQA |0.5 | 1 |
|Hyperparisan |0.01	|0.01|

## Cite

If you use the code, please cite this paper:

```
@inproceedings{ye2021trbert,
  author    = {Deming Ye and
               Yankai Lin and
               Yufei Huang and
               Maosong Sun},
  editor    = {Kristina Toutanova and
               Anna Rumshisky and
               Luke Zettlemoyer and
               Dilek Hakkani{-}T{\"{u}}r and
               Iz Beltagy and
               Steven Bethard and
               Ryan Cotterell and
               Tanmoy Chakraborty and
               Yichao Zhou},
  title     = {{TR-BERT:} Dynamic Token Reduction for Accelerating {BERT} Inference},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               {NAACL-HLT} 2021, Online, June 6-11, 2021},
  pages     = {5798--5809},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://doi.org/10.18653/v1/2021.naacl-main.463},
  doi       = {10.18653/v1/2021.naacl-main.463},
  timestamp = {Fri, 06 Aug 2021 00:41:32 +0200},
  biburl    = {https://dblp.org/rec/conf/naacl/YeLHS21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
