from llava.train.train_suit import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")