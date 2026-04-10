This project is a two-model speech correction pipeline built for people with atypical speech conditions like dysarthria and Down syndrome. You speak into the app, your audio runs through a fine-tuned Whisper model that converts it to text, and then a fine-tuned Flan-T5 grammar correction model cleans up the transcript into proper English. The corrected sentence is spoken back to you using the Web Speech API.

Hardware required: a GPU with at least 8GB VRAM for training. The scripts were run on an NVIDIA RTX 4060. CPU-only machines will still work but training will take much much longer. Software used: Python 3.11, VS Code.

How to run it:

1. Install dependencies: pip install transformers datasets accelerate librosa soundfile evaluate torch sentencepiece sacrebleu flask flask-cors gunicorn
2. Download the TalkBank corpora (.cha files) for scripts 6 and 7. The datasets used are Flusberg, Rollins, Eigsti, AAC, QuigleyMcNally, NYU-Emerson, Nadig, and AFG. These are available through TalkBank at https://talkbank.org. They are also attached to the github.
3. If you want to retrain from scratch, run the scripts in order: 1 through 8. Before you run anything, change all your paths to your paths for where you want the things to go as they are right now hard coded directories. Script 1 downloads the TORGO audio data from https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html. Script 2 parses it into pairs. Script 3 fine-tunes Whisper which is downloaded from openai/whisper-small from HuggingFace automatically. Script 4 tests the fine-tuned Whisper model by transcribing audio. Script 6 parses the TalkBank .cha files. Script 7 fine-tunes the grammar model which incolves downloading google/flan-t5-base from HuggingFace automatically and also downloading the JFLEG dataset. Script 8 takes a raw transcript and returns a grammatically corrected version while checking that the original meaning is preserved. The model weights below are proof that the code works: [https://huggingface.co/PINEAPPLEANANAS/SpeechBoeing](https://huggingface.co/PINEAPPLEANANAS/SpeechBoeing)

Reference links used for implementation for the harder parts (IDK if this is supposed to go here but here it is):

- Hugging Face training loop and Seq2SeqTrainer:
  - https://huggingface.co/docs/transformers/training
  - https://huggingface.co/docs/transformers/main_classes/trainer
  - https://huggingface.co/learn/nlp-course/chapter3/3
- Whisper fine-tuning:
  - https://huggingface.co/blog/fine-tune-whisper
- Grammar error correction and seq2seq:
  - https://huggingface.co/learn/nlp-course/chapter1/1
  - https://paperswithcode.com/task/grammatical-error-correction
- BLEU score and evaluation metrics:
  - https://huggingface.co/spaces/evaluate-metric/bleu
  - https://www.nltk.org/api/nltk.translate.bleu_score.html
- Gradient accumulation and checkpointing:
  - https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one
- DataCollator and padding:
  - https://huggingface.co/docs/transformers/main_classes/data_collator
- Python dataclasses:
  - https://docs.python.org/3/library/dataclasses.html
  - https://realpython.com/python-data-classes
- SequenceMatcher for string comparison:
  - https://docs.python.org/3/library/difflib.html
- Librosa for audio processing:
  - https://librosa.org/doc/latest/tutorial.html
 
Some Python standards used in these files: print(f"...") is the correct way to insert variables into strings in Python rather than concatenating with plus signs. Type hints, dataclasses, and context managers are also used throughout, which are standard modern Python 3 patterns.
