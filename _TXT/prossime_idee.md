# Piani futuri per il Transformer:
## Fase 1:
- [ ] spiegare bene il codice. Spiegare bene cosa fanno i layer e come si usano tramite
  docstring. Commentare tutte le dimensioni;
- [ ] rifinire il codice. Introdurre della logica che fonde i layer qkv nella self attention. Verificare quando c'e' bisogno della normalization a fine encoder-decoder (probabilmente legato a norm_first). Spulcia l'implementazione di Torch per notare altri accorgimenti simili (e chiedili a GPT che li sa);
- [ ] ottimizzazione. Prendi le funzioni forward (anche del tensore) e vedi se puoi
  riscriverle per velocizzarle o renderle piu' chiare;
- [ ] creare un tokenizer. Un byte-pair per il momento e' piu' che sufficiente;
- [ ] introdurre la mixed-precision (occhio alle promozioni)
- [ ] implementare grafici per tracking loss e scores vari
- [x] rifare il training e vedere come si comporta
- [x] implementa la visualizzazione degli attn_scores (dovresti essere in grado
  di riciclare la logica usata per le confusion matrix)
- [ ] fixxa questione .broadcast_to() in backward
- [ ] implementa dei seed per la riproducibilita', cosi' capisci che impatto
  hanno le tue modifiche. Usa il seed per inizializzare i pesi e per fare lo shuffle
- [ ] aggiusta l'inconsistenza con il PositionalEncoding
+ [ ] one time print di debug per segnalare qual e' il picco di streams usate
  simultaneamente (memorizza la lunghezza minore di available streams o qualcosa cosi') e
  dove decide di parallelizzare con quale valutazione (fancy print del dict)
- [ ] visualizzazione embeddings
- [x] convertire la classe Transformer come child di Module (TransformerV2)
- [ ] e' vero che talvolta conviene ricalcolare le attivazioni piuttosto che salvarle
  per la backward? => Implementa gradient checkpointing
- [ ] conviene salvarsi le activation functions per la backward o ricalcolarle in bw
  per risparmiare memoria?

## Fase 2:
- [ ] implementare nuove act funct (Llama 3.1 usa swiGLU);
- [ ] introdurre beam search e altri algoritmi per prendere il prossimo token
- [ ] iniziare ad apportare migliorie all'architettura per orientarlo verso un
  Transformer++ decoder-only;
- [ ] implementare i RoPE;
- [ ] implementare i MoE? La memoria lo permette?
- [ ] cercare nuovi tipi di attenzione e migliorie da apportare;
- [ ] iniziare a costruire una piccola struttura per usarlo (mi piacerebbe una bella
  finestra shell stile minigame di Snow White con effetti di Neural Cloud e *colori*)
- [ ] iniziare a mettere da parte un dataset buono;
- [ ] iniziare ad implementare un tokenizer SOTA (TikToken?)
- [ ] fare training di test;
- [ ] "needle in a haystack" benchmark;
- [ ] implementare AdamW/Lion (piu' leggero di Adam) o AdaFactor
- [ ] implementa la curva della scaling law per vedere come si comporta il tuo modello
- [ ] cos'e' la grouped multi query attention con caching qk o kv? Vale la pena implementarla?



## Molto futuro:
- [ ] provare ad usare il modello (e quindi la libreria) su telefono (**qua** segna cio' che hai imparato)
- [ ] scrivere le backward piu' pesanti sottomoforma di RawKernel

## Libreria:
- [ ] fix load ModuleList
+ [ ] one time print di debug per segnalare qual e' il picco di streams usate
  simultaneamente (memorizza la lunghezza minore di available streams o qualcosa cosi') e
  dove decide di parallelizzare con quale valutazione (fancy print del dict)
- [ ] prendi spunto da ollama se vuoi far girare i tuoi modelli su mobile

https://www.firebird-technologies.com/p/how-to-make-more-reliable-reports