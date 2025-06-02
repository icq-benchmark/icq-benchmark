import time
import json
import openai
import difflib
import string
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback

# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback
from openai import OpenAI
from openai import RateLimitError, APIConnectionError
import os
import random

from .models import ModelLoader


class NegCaptionGenerator:
    def __init__(self, model_name: str, device: str = 'cpu', use_chatgpt=False):
        self.model_name = model_name
        self.device = device
        self.use_chatgpt = use_chatgpt
        self.max_retries = 10
        self.retry_delay = 60

        # if not self.use_chatgpt:
        #     self.model, self.processor = self.load_model(model_name, device)

        if self.use_chatgpt:
            self.client = self.load_client()
        else:
            self.model, self.processor = self.load_model(model_name, device)

    def load_model(self, model_name, device):
        if model_name == '':
            print("No model_name provided. Loading default model.")
            model_name = 'llava'
        return ModelLoader(model_name, device)

    def load_client(self):
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            # os.getenv("OPENAI_API_KEY")
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        return client

    def generate_chatgpt(self, prompt):
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",  # "gpt-35-turbo-16k",
                    messages=[
                        {
                            "role": "system",
                            "content": ""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=250,
                    temperature=0.7,
                    top_p=1.0,
                    n=1,
                    stop=None
                )
                return response.choices[0].message.content
            except (RateLimitError, APIConnectionError, ValueError) as e:
                retries += 1
                print(
                    f"Attempt {retries} failed: {str(e)}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

                if retries >= self.max_retries:
                    print(f"Max retries exceeded. Exiting.")
                    return None

    def generate_random_neg_caption(self, positive_caption, prompt):
        selected_prompt_key, selected_prompt_template = random.choice(
            list(prompt.items()))

        neg_prompt = selected_prompt_template.format(positive_caption)

        negative_caption = {}
        refinements = {}

        if self.model_name == "gpt-35-turbo-16k":
            neg_caption = self.generate_chatgpt(neg_prompt)
        else:
            if self.processor.pad_token is None:
                self.processor.pad_token = self.processor.eos_token

            inputs = self.processor(
                neg_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

            generate_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.processor.pad_token_id,
                max_new_tokens=250
            )

            neg_caption = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        neg_caption = self.postprocess_neg_caption(
            neg_caption, positive_caption)

        negative_caption[selected_prompt_key] = neg_caption

        refined_words = self.get_refinements(positive_caption, neg_caption)
        refinements[selected_prompt_key] = refined_words

        return negative_caption, refinements

    def generate_neg_caption(self, positive_caption, prompt):
        negative_caption = {}
        refinements = {}

        for key, neg_prompt_template in prompt.items():
            neg_prompt = neg_prompt_template.format(positive_caption)
            if self.model_name == "gpt-35-turbo-16k":
                # neg_caption = self.generate_with_chatgpt(neg_prompt, entry_index, result_index)
                neg_caption = self.generate_chatgpt(neg_prompt)

            else:
                if self.processor.pad_token is None:
                    self.processor.pad_token = self.processor.eos_token
                inputs = self.processor(
                    neg_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.processor.pad_token_id,
                    max_new_tokens=250
                )
                neg_caption = self.processor.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            neg_caption = self.postprocess_neg_caption(
                neg_caption, positive_caption)

            negative_caption[key] = neg_caption

            refined_words = self.get_refinements(positive_caption, neg_caption)
            refinements[key] = refined_words

        return negative_caption, refinements

    def get_refinements(self, pos_caption, neg_caption):
        pos_caption_clean = self.remove_punctuation(pos_caption).split()
        neg_caption_clean = self.remove_punctuation(neg_caption).split()

        matcher = difflib.SequenceMatcher(
            None, pos_caption_clean, neg_caption_clean)
        refinements = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                pos_words = ' '.join(pos_caption_clean[i1:i2])
                neg_words = ' '.join(neg_caption_clean[j1:j2])
                refinements.append(f'change "{pos_words}" to "{neg_words}"')
            elif tag == 'delete':
                pos_words = ' '.join(pos_caption_clean[i1:i2])
                refinements.append(f'remove "{pos_words}"')
            elif tag == 'insert':
                neg_words = ' '.join(neg_caption_clean[j1:j2])
                refinements.append(f'add "{neg_words}"')

        return refinements

    def postprocess_neg_caption(self, generated_text, original_caption):
        if generated_text is None:
            generated_text = ''

        cleaned_text = generated_text.replace(original_caption, "").strip()
        sentences = cleaned_text.split(". ")
        unique_sentences = list(dict.fromkeys(sentences))

        neg_caption = ". ".join(unique_sentences)

        if "Answer:" in neg_caption:
            neg_caption = neg_caption.split("Answer:")[1].strip()
        if "Output:" in neg_caption:
            neg_caption = neg_caption.split("Output:")[1].strip()
        if '\n' in neg_caption:
            neg_caption = neg_caption.split('\n')[0]

        return neg_caption

    def remove_punctuation(self, text):
        return ''.join([char for char in text if char not in string.punctuation])

    def process_dataset(self, data, prompt):
        processed_data = []

        data = data.get('image-caption', [])
        print(f"Processing {len(data)} entries from the dataset.")

        for entry_index, entry in enumerate(data):

            img_caption = entry
            results = img_caption.get('results', [])
            print(
                f"Processing {len(results)} results from entry {entry_index}.")

            for result_index, result in enumerate(results):

                positive_caption = result.get('positive_caption', "")
                original_caption = result.get('original_caption', "")

                negative_caption, refinements = self.generate_neg_caption(
                    positive_caption, prompt, entry_index, result_index)
                if not negative_caption:
                    print(
                        f"Skipping entry {entry_index}, result {result_index} due to error.")
                    continue

                entry_data = {
                    'image_id': result.get('image_id'),
                    'original_caption': original_caption,
                    'positive_caption': positive_caption,
                    'negative_caption': negative_caption,
                    'refinements': refinements
                }

                processed_data.append(entry_data)

        if processed_data:
            print(f"Processed {len(processed_data)} valid results.")
        else:
            self.reset_progress()
        return processed_data

    def timer(self, chain, text, max_retries=5, retry_delay=5):
        retries = 0
        start_time = time.time()

        while retries < max_retries:
            try:
                output = chain({"text": text})
                end_time = time.time()
                elapsed_time = end_time - start_time

                return output, elapsed_time

            except (RateLimitError, APIConnectionError, ValueError) as e:
                retries += 1
                print(
                    f"Attempt {retries} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

                if retries >= max_retries:
                    print(f"Max retries exceeded. Exiting.")
                    return None, None

    def generate_chatgpt_langchain(self, prompt):
        llm = OpenAI(model_name="gpt-3.5-turbo-16k",
                     openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

        template = """In this task, you are given an input sentence. :  
        ''' {text}'''
        """

        prompt_template = PromptTemplate(
            input_variables=["text"], template=template)

        chain = LLMChain(llm=llm, prompt=prompt_template, output_key="Output")
        overall_chain = SequentialChain(chains=[chain], input_variables=[
                                        "text"], output_variables=["Output"])

        with get_openai_callback() as cb:
            output, exec_time = self.timer(overall_chain, prompt)

            token_used = cb.completion_tokens

            if output:
                print(f"Output: {output['Output']}")
            print(f"Execution Time: {exec_time} seconds")
            print(f"Completion Tokens Used: {token_used}")

            return output
