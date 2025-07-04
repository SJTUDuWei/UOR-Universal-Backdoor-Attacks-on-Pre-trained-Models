# UOR-Universal-Backdoor-Attacks-on-Pre-trained-Models

This is the implementation of our paper 'UOR：Universal Backdoor Attacks on Pre-trained Models', accepted by the ACL-Findings 2024.


## Requirements
- python == 3.9.18
- torch == 2.1.1 (cuda12.1)
- transformers == 4.21.3  
- datasets == 2.15.0  
- openprompt == 1.0.1  
- seqeval == 1.2.2
- wordfreq == 3.1.1
- umap-learn == 0.5.5
- matplotlib



## Implementation
We have taken reference from the [Openbackdoor](https://github.com/thunlp/OpenBackdoor) implementation but removed the attacker from it.

The main program includes 
- attack_plm.py : search triggers + attack pre-trained model + fine-tune downstream tasks
- grad_search.py : search triggers only 
- ft_sc.py : fine-tuning Sequence Classification tasks only 
- pt_sc.py : prompt-tuning Sequence Classification tasks only 
- dsm_test.py : test the clean/poisoned plm on downstream tasks only 


## Run
We use shell scripts to run the code. For example, we attack the bert model with 6 triggers through the script shown below.
```
CUDA_VISIBLE_DEVICES=0 python attack_plm.py --config_path ./configs/uor/bert_trigger6.yaml
```
The '.yaml' file shows the detailed parameters.


## Citation
```
@inproceedings{du2024uor,
  title={UOR: Universal Backdoor Attacks on Pre-trained Language Models},
  author={Du, Wei and Li, Peixuan and Zhao, Haodong and Ju, Tianjie and Ren, Ge and Liu, Gongshen},
  booktitle={Findings of the Association for Computational Linguistics ACL 2024},
  pages={7865--7877},
  year={2024}
}
```
