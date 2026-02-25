import pandas as pd
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'mistralai/Mistral-Small-3.2-24B-Instruct-2506'
)

# Load guidelines
with open('data/preprocessing/annotation_guidelines_v1.md', 'r', encoding='utf-8') as f:
    guidelines = f.read()

# Load data
df = pd.read_csv('data/preprocessing/test_set_mistral.csv', sep=';', encoding='utf-8')

# Token counts
n_guidelines = len(tokenizer.encode(guidelines))
n_headline   = df['headline'].dropna().apply(lambda x: len(tokenizer.encode(str(x))))
n_content    = df['content'].dropna().apply(lambda x: len(tokenizer.encode(str(x))))
n_paragraph  = df['paragraph_text'].dropna().apply(lambda x: len(tokenizer.encode(str(x))))
n_sentence   = df['sentence_text'].dropna().apply(lambda x: len(tokenizer.encode(str(x))))

print(f'Guidelines:      {n_guidelines} tokens')
print(f'Headline   mean={n_headline.mean():.0f}  max={n_headline.max()}')
print(f'Content    mean={n_content.mean():.0f}  max={n_content.max()}')
print(f'Paragraph  mean={n_paragraph.mean():.0f}  max={n_paragraph.max()}')
print(f'Sentence   mean={n_sentence.mean():.0f}  max={n_sentence.max()}')

# Total estimate (mean and max)
mean_total = n_guidelines + n_headline.mean() + n_content.mean() + n_paragraph.mean() + n_sentence.mean()
max_total  = n_guidelines + n_headline.max()  + n_content.max()  + n_paragraph.max()  + n_sentence.max()

print(f'')
print(f'Total (mean case): {mean_total:.0f} tokens')
print(f'Total (worst case): {max_total} tokens')
print(f'+ output (512):    mean={mean_total+512:.0f}  max={max_total+512}')