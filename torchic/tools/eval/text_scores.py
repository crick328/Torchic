import math


# metterei tutti gli scores in un file a parte
def bleu_score(pred_sentence: str, trg_sentence: str, max_ngram_len: int=4) -> float:
    def extract_ngrams(words, ngram_len):
        '''
        Returns the sentence divided in n-grams of ngram_len.
        '''
        return [tuple(words[i:i+ngram_len]) for i in range(len(words) - ngram_len + 1)]
    
    def calc_clipped_precision(n: int):
        '''
        Returns the clipped precision of the given translation using
        n-grams of len n.
        '''
        # suddivido i tokens in n-grams (sequenze di n tokens consecutivi)
        # NOTE: l'n-gram successivo include l'ultimo token dell'n-gram
        # precedente (se n>1 ovviamente)
        pred_ngrams = extract_ngrams(pred_sentence, n)
        trg_ngrams = extract_ngrams(trg_sentence, n)

        # il bleu conta gli n-grams che matchano per calcolare lo score. Pero',
        # e' importante che il punteggio non venga inflazionato da n-grams generati
        # erroneamente piu' volte (trg: ["ciao", "Manu"], pred: ["ciao", "ciao", "ciao"].
        # Non devo dare il triplo dei punti perche' ci sono 3 n-grams che matchano)

        # prendo gli n-grams generati dal modello e non quelli della traduzione
        # in modo da avere gia' un count di 0 per quelli non predetti
        unique_ngrams = set(pred_ngrams)
        clipped_counts = {}

        for ngram in unique_ngrams:
            # per ogni n-gram unico, conto le sue occorrenze in entrambe le
            # frasi e memorizzo il count piu' basso (per evitare che prediction
            # sbagliate inflazionino il punteggio). Se la rete ha lavorato bene,
            # i due minimi coincidono, altrimenti il count nella frase predetta e'
            # (erroneamente) piu' elevato
            clipped_counts[ngram] = min(pred_ngrams.count(ngram), trg_ngrams.count(ngram))

        return sum(clipped_counts.values()) / max(1, len(pred_ngrams))
    

    # calcolo la clipped precision per ogni lunghezza di n-gram, da
    # 1 a max_ngram_len
    precisions = []
    for i in range(1, max_ngram_len+1):
        precisions.append(calc_clipped_precision(i))

    pred_len = len(pred_sentence)
    trg_len = len(trg_sentence)

    # non dovrebbe succedere, ma non si sa mai
    if pred_len == 0:
        return 0.0

    if pred_len > trg_len:
        bp = 1
    else:
        bp = math.exp(1 - trg_len/pred_len)

    # Geometric Average Precision
    gap = 1
    for pn in precisions:
        gap *= pn**(1/4)
    
    return bp * gap