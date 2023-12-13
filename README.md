# Answer-Aware Question Generation: A Comparative Study with Google T5 Transformers and Meta’s Llama2

![Llama2 Logo](report/llama-coding.png)

## Abstract

This report explores the capabilities of Transformers, especially Sequence To Sequence Large Language models in the task of Answer-Aware question generation. For this project's purposes, we decided to employ Google T5 model for English conditional text generation, in different sizes and with different language modelling approaches. The experimental outcome consistently confirmed the state of the art results obtaining up to 0.92 BERTScore (F1) on T5-Base Transformer (223M parameters) and 41.1 ROUGE-1 with supervised fine tuning. We also assessed Causal Language Modelling performance on Meta's Llama 2, obtaining similar metrics as sequence-to-sequence models, but producing more variegate and accurate questions. Finally we employed Proximal Policy Optimization Reinforcement Learning techniques to lexically and semantically improve models results. The good outcome of this analysis again confirms LLMs helpfulness and power in solving language modelling tasks, that is crucial for practical applications, especially in educative and e-learning settings.

## Repository Structure

- **scripts**: Contains the code for supervised fine-tuning of models and evaluation.
- **notebooks**: Includes Jupyter notebooks used for quick code runs or exploration.
- **configs**: Holds DeepSpeed configuration files for parallel Zero3 model training.
- **results**: Stores the results of your experiments.
- **report**: Contains the LaTeX report for your project.

## Dependencies

This projects mainly relies on the following libraries:
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [DeepSpeed](https://www.deepspeed.ai/)

### Install Dependencies
Straightforward installation of the dependencies can be done with the following command:

```bash
pip install -r requirements.txt
```

## How to Run
In case you want to run the code, you can follow the steps below. You can also run the code in the notebooks folder.

### Fine-tuning
```bash
cd scripts
python t5-trainer.py --model_name [modelname] --batch_size [batchsize] 
```

### Evaluation
```bash
cd scripts
python t5-evaluate.py --model_name [modelname] 
```
For other options, please refer to the scripts folder and run scripts with `--help` command.


## Configuration
This folder contains deepspeed configuration files for parallel training. In case you want to run the code in parallel, you can configure accelerate and deepspeed with the following command:

```bash
accelerate config
```
See Accelerate [documentation](https://huggingface.co/docs/accelerate/index) for more information.



