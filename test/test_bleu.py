from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from test_train_transformer import bleu_score

# Frase generata
predicted_tokens = ["the", "cat", "sat", "on", "the", "mat"]

# Frase target
trg_tokens = ["the", "cat", "is", "on", "the", "mat"]

# Per calcolare BLEU a livello di frase
bleu = sentence_bleu([trg_tokens], predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25))  # Uniform weights
print(f"BLEU score (NLTK): {bleu}")

my_bleu = bleu_score(predicted_tokens, trg_tokens)
print(f"My BLEU: {bleu}")
