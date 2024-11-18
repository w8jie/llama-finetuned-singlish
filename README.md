# llama-finetuned-singlish

## Introduction
Current existing chat LLMs struggle to maintain its defined chat persona (i.e, reply in pirate language) over an extended amount of time.

The project aims to finetune llama-3.2-3B-Instruct with singlish conversation pairs such that it maintains a Singlish chat persona.

## Data Generation
10K conversational QA pairs were synthetically generated using GPT-4o containing English questions with Singlish answers.
| index | singlish (Qn) | english (Ans) |
| --- |------| ---|
| 1 | How do you make chicken rice from scratch? | Wah, chicken rice ah? First must boil the chicken, then cook the rice with chicken stock. Don't forget the chilli and ginger sauce, very important one. |
| 2 | How do you stay motivated to study for long hours? | Wah, sometimes really hard leh, but I take short breaks and reward myself with snacks lor. |
| 3 | Do you usually have trouble falling asleep at night? | Sometimes lah, especially when got a lot on my mind. |

- Full trianing data: `english_singlish_chat_v0.2.csv`
- Data generation script: `data_generation.py`


## Running the model
1. Create virtual env `python3 -m venv /path/to/new/virtual/environment`
2. Install requirements `pip install -r requirements.txt`
3. Load and run model through notebook `main.ipynb`
