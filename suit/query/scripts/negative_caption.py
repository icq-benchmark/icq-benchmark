from models import ModelLoader
import difflib
import string


class NegCaptionGenerator():
    def __init__(self, dataset: str, model_name: str, device: str = 'cpu'):
        self.dataset = dataset
        self.model_name = model_name
        self.device = device
        self.model, self.processor = self.load_model(model_name, device)

    def load_model(self, model_name, device):
        if model_name == '':
            print("No model_name provided. Loading default model.")
            model_name = 'llava'
        return ModelLoader(model_name, device)

    def generate_neg_caption(self, positive_caption, prompt):
        negative_caption = {}
        refinements = {}

        for key, neg_prompt_template in prompt.items():
            neg_prompt = neg_prompt_template.format(positive_caption)

            # Set pad_token as eos_token if pad_token is not available
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
            print(f"Generated negative caption for {key}: {neg_caption}")

            if "Answer:" in neg_caption:
                neg_caption = neg_caption.split("Answer:")[1].strip()
            if "Output:" in neg_caption:
                neg_caption = neg_caption.split("Output:")[1].strip()
            if '\n' in neg_caption:
                neg_caption = neg_caption.split('\n')[0]
            negative_caption[key] = neg_caption
            print(f"\n Negative caption for {key}: {neg_caption}\n")

            refined_words = self.get_refinements(positive_caption, neg_caption)
            refinements[key] = refined_words

        return negative_caption, refinements

    def remove_punctuation(self, text):
        """
        Remove punctuation from the given text.
        """
        return ''.join([char for char in text if char not in string.punctuation])

    def get_refinements(self, pos_caption, neg_caption):
        """
        Identify the changes made between the positive and negative captions
        and return the refinements in the form of 'change "X" to "Y"', ignoring punctuation.
        """
        # Clean punctuation
        pos_caption_clean = self.remove_punctuation(pos_caption).split()
        neg_caption_clean = self.remove_punctuation(neg_caption).split()

        # Use difflib to get matching and differing segments
        matcher = difflib.SequenceMatcher(
            None, pos_caption_clean, neg_caption_clean)
        refinements = []

        # Iterate over matching/diffing blocks
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':  # Words have been changed
                pos_words = ' '.join(pos_caption_clean[i1:i2])
                neg_words = ' '.join(neg_caption_clean[j1:j2])
                refinements.append(f'change "{pos_words}" to "{neg_words}"')
            elif tag == 'delete':  # Words removed from positive caption
                pos_words = ' '.join(pos_caption_clean[i1:i2])
                refinements.append(f'remove "{pos_words}"')
            elif tag == 'insert':  # Words added in negative caption
                neg_words = ' '.join(neg_caption_clean[j1:j2])
                refinements.append(f'add "{neg_words}"')

        return refinements

    def postprocess_neg_caption(self, generated_text, original_caption):
        cleaned_text = generated_text.replace(original_caption, "").strip()
        sentences = cleaned_text.split(". ")
        unique_sentences = list(dict.fromkeys(sentences))
        return ". ".join(unique_sentences)

    def process_dataset(self, data, dataset, prompt):
        processed_data = []
        if dataset == "circo":
            circo_data = data.get('circo-val', data.get('circo-test', []))

            for entry_index, entry in enumerate(circo_data):
                img_caption = entry.get('image-caption', {})
                results = img_caption.get('results', [])
                print(f"Processing {len(results)} entries from the dataset.")

                for result_index, result in enumerate(results):
                    positive_caption = result.get('positive_caption', {})
                    negative_caption, refinements = self.generate_neg_caption(
                        positive_caption, prompt)
                    if not negative_caption:
                        print(
                            f"Skipping entry {entry_index}, result {result_index} due to missing semantic dimensions.")
                        continue

                    entry_data = {
                        'img_id': result.get('img_id'),
                        'positive_caption': positive_caption,
                        'negative_caption': negative_caption,
                        'refinements': refinements
                    }
                    processed_data.append(entry_data)

            print(f"Processed {len(processed_data)} valid results.")

        return processed_data
