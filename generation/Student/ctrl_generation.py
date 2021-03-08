import torch
import pandas as pd
import nltk
nltk.download('punkt')
from nltk import tokenize
import logging
from tqdm import tqdm
from tqdm import trange
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

def prepare_ctrl_input(temperature, _, tokenizer, prompt_text):
    if temperature > 0.7:
        print ("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        print ("WARNING! You are not starting your generation from a control code so you won't get good results")
    return encoded_prompt#prompt_text


def prepare_xlm_input(xlm_language, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if xlm_language in available_languages:
            language = xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(padding_text, _, tokenizer, prompt_text):
    prompt_text = (padding_text if padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(padding_text, _, tokenizer, prompt_text):
    prompt_text = (padding_text if padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}

MAX_LENGTH = int(10000)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


device = torch.device("cuda:1" if torch.cuda.is_available()  else "cpu")

def initialize_model(model_type, length, device):

  MODEL_CLASSES = {
      "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
      "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
      "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
      "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
      "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
      "xlm": (XLMWithLMHeadModel, XLMTokenizer),
  }
  try:
      model_class, tokenizer_class = MODEL_CLASSES[model_type]
  except KeyError:
      raise KeyError("the model {} you specified is not supported.")
  tokenizer = tokenizer_class.from_pretrained(model_type)
  model = model_class.from_pretrained(model_type)
  model.to(device)
  length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)
  return model, tokenizer, length

def generate_text_from_condition(model, tokenizer, length, prompt_text, repetition_penalty, temperature, num_return_sequences, model_name, stop_token = None, model_type = "ctrl"):
        device = model.device
       # Initialize the model and tokenizer
        # prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

        # Different models need different input formatting and/or extra arguments
        print ("o")
        requires_preprocessing = model_type in PREPROCESSING_FUNCTIONS.keys()
        print ("1")
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(model_type)
            preprocessed_prompt_text = prepare_input(temperature, model, tokenizer, prompt_text)
            encoded_prompt = tokenizer.encode(
              preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", add_space_before_punct_symbol=True
            )
        else:
            encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        print ()
        encoded_prompt = encoded_prompt.to(device)
        print ("2")
        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt
        print ("3", model.device, input_ids.device)
        output_sequences = model.generate(
          input_ids=input_ids,
          max_length=length + len(encoded_prompt[0]),
          temperature=temperature,
          # top_k=args.k,
          # top_p=args.p,
          repetition_penalty=repetition_penalty,
          do_sample=True,
          num_return_sequences=num_return_sequences,
        )
        print ("4", output_sequences)
        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

          # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

          # Remove all text after the stop token
            text = text[: text.find(stop_token) if stop_token else None]

          # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )
            if model_name == 'ctrl':
                print ("ctrl")
                total_sequence = total_sequence.replace(" \n \n ", "")
                total_sequence = total_sequence.replace("  ", "")
                total_sequence = total_sequence.replace("    ", "")
                total_sequence = total_sequence.replace("     ", "")
                total_sequence = " ".join(tokenize.sent_tokenize(total_sequence)[:-1])
            else:
                total_sequence = total_sequence.replace("\n", "")
                total_sequence = total_sequence.replace("  ", "")
                total_sequence = total_sequence.replace("    ", "")
                total_sequence = total_sequence.replace("     ", "")
                total_sequence = " ".join(tokenize.sent_tokenize(total_sequence)[:-1])
            total_sequence = total_sequence.replace(" \n \n ", "")
            print ("total_sequence", total_sequence)
            generated_sequences.append(total_sequence)
        return generated_sequences