"""
For baseline exps: Captioning. Use GPT to adjust the image captioning
of the generated images.
"""
import argparse
import openai
from openai import OpenAI
from utils import load_jsonl, save_file, remove_extra_spaces

API_KEY = 'YOUR_API_KEY'
client = OpenAI(api_key=API_KEY, max_retries=10)
rewrite = False


def _transform_caption(input_data, change_type, modified_detail,
                       original_detail):
    prompt = f"""
    I have a caption "{input_data}", adjust the {change_type} from {modified_detail} to {original_detail}.
    The revised caption should remain coherent and logical without introducing
    any additional details.
    """
    return prompt

def _transform_caption_scribble(input_data, new_type, new_detail):
    prompt = f"""
    I have a caption "{input_data}". Modify it by adding {new_type}
    "{new_detail}". The revised caption should remain coherent and logical
    without introducing any other additional details.
    """
    return prompt


def _manipulate(prompt):
    # try requesting until we get a response instead of a rate limit error
    while True:
        try:
            completions = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=50,
                temperature=0.1
            )
            break
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(
                e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)

    message = completions.choices[0].text
    return message


def generate_caption(original_caps, anno_details, is_scri):
    # to ensure the captions' length consistent with our filtered annotations
    new_original_caps = [item1 for item2 in anno_details for item1 in
                         original_caps if
                         item1.get("qid") == item2.get("qid")]
    new_data = []
    for idx, data in enumerate(new_original_caps):
        query = data['query']
        anno_detail = anno_details[idx]
        assert data['qid'] == anno_detail['qid']
        if is_scri:
            if anno_detail['has_new_detail']:
                new_types = anno_detail['new_detail_type']
                for i in range(len(new_types)):
                    new_type = anno_detail['new_detail_type'][i]
                    new_detail = anno_detail['added_details'][i]
                    prompt = _transform_caption_scribble(query, new_type, new_detail)
                    prompt = remove_extra_spaces(prompt)
                    new_caption = _manipulate(prompt)
                    new_caption = remove_extra_spaces(new_caption)
                print(data['query'], " TO ", new_caption)
                new_data.append({'qid': data['qid'], 'query': new_caption})
                break
        else:
            if anno_detail['has_modification']:
                mod_type = anno_detail['modification_type'][0]
                fake_detail = anno_detail['modified_details'][0]
                mod_detail = anno_detail['removed_details'][0]

                prompt = _transform_caption(query, mod_type, fake_detail,
                                            mod_detail)
                prompt = remove_extra_spaces(prompt)
                new_caption = _manipulate(prompt)
                new_caption = remove_extra_spaces(new_caption)
                print(idx, anno_detail['old_query'])
                print(query, " TO ", new_caption)
                new_data.append({'qid': data['qid'], 'query': new_caption})
                break
            else:
                new_data.append({'qid': data['qid'], 'query': query})
                print(f"No modification is made to {data['qid']}")
    print("New captions are generated.")
    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate and save modified captions.")
    parser.add_argument('--ref_img_cap_file', type=str, required=True,
                        help="Path to the reference image captions file.")
    parser.add_argument('--gt_file', type=str, required=True,
                        help="Path to the ground truth file.")
    parser.add_argument('--is_scribble', action='store_true',
                        help="Whether the reference image style is scribble or not.")
    parser.add_argument('--des_path', type=str, required=True,
                        help="Path to save the refined captions.")

    args = parser.parse_args()

    ref_img_caps = load_jsonl(args.ref_img_cap_file)
    annotations = load_jsonl(args.gt_file)
    refined_caps = generate_caption(ref_img_caps, annotations, args.is_scribble)
    save_file(args.des_path, refined_caps)
