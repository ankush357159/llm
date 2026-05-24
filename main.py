from embedding.SimpleTokenizer import SimpleTokenizer
def main():
    print("Hello from llm!")
    
def run():
    tokenizer = SimpleTokenizer()
    sample = "Hello world! This is clean code."
    print(tokenizer.tokenize(sample))


if __name__ == "__main__":
    main()
    run()
