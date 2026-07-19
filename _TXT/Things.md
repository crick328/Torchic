Voglio riordinare e riassumere tutto cio' che ho scritto nei txt vari

# Cache
Cose a cui stavo lavorando:
- PROBLEMONE: l'approccio attuale di AMP e' un vicolo cieco. Dovrei decorare tutte le funzioni, ma avrei un overhead mostruoso e il wrapper sarebbe attivo anche ad AMP spento. In piu', pytorch funziona a livello di singola operazione, mentre il mio approccio wrappa funzioni grosse. Potrei creare una chiamata aggiuntiva (non wrapper) in tutte le funzioni elementari di Tensor, ma cosi' tutte le forward che usano .data per staccarsi dal grafo non potrebbero mai ricevere il trattamento di AMP. Tutto cio' mi puzza di 3.2: cancelliamo l'uso dei .data (che ci sta, cosi' mi assicuro che tutto funzioni come dico io), modernizziamo AMP e creiamo della logica per far si che i forward con backward custom vengano trattati correttamente (non posso usare no_grad o spacco tutto). Prendila come versione sperimentale perche' non so che danni posso creare a livello di overhead con tutti sti tensori intermedi. **Insomma**, per ora o devio per sfruttare al massimo AMP e spammare ovunque di tensori (insostenibile) o 
- => pytorch non lavora esattamente a livello di singola operazione, ma di **primitiva**. Ad esempio, std() e' un'operazione che e' composta da tanti calcoli intermedi. Tuttavia, AMP non li vede perche' std e' implementata come operazione a basso livello, in C++. AMP intercetta solo i tensori python.
Tornando al _tuo_ AMP, per ora potrebbe pure funzionare: gli input sono castati dinamicamente e potrestio decorare solo le operazioni che sono in whitelist perche' tanto gli output sono castati automaticamente a float32 quando incapsuli il risultato nel Tensor.
  - L'unico problema di questo approccio e' che cosi' crei un rimbalzo continuodi casting se ad esempio hai tre Linear di fila... Posso lavorarci sopra comunque.
  - Un altro problema e' che le firme del wrapper devono essere giuste perche' args e kwars sono _lentissimi_ e bruciano memoria. Posso fare due decoratori che condividono la logica e hanno firme per funzioni a singoli arg e doppio, ma non mi piace come soluzione.
  - Piuttosto, vorrei sfruttare di piu' il fatto che i decoratori hanno una porzione di logica che parte **all'avvio della libreria**.
  - La questione delle copie in fp16 si risolve in fretta, ma non mi piace. Dovrei tenere una copia fissa in memoria, ma allora quale sarebbe il vantaggio?

- nuova idea per AMP: il decortore servira' solo per dire "ehi, controlla se quest'operazione e' safe" e sara' fatto all'avvio. Non avra' nient'altro, quindi zero overhead in runtime. Volendo si puo' anche creare una cache: se l'operazione non e' presente in questo file AMP_cache.json, parte is_safe_in_float16, senno' legge la classificazione (volendo si possono creare impostazioni per resettarla e cosi' via).  Quando si entra nel context, scopo
- **LE ACTIVATION_FUNCTION HANNO IL __dict__ PERCHE" NON UYSANO GLI SLOTS!!** Ci sono altri casi del genere??
- manca ToBackward


# Next
- Grad Checkpointing
- AMP (https://docs.pytorch.org/docs/2.12/amp.html#module-torch.amp)
- In-place Ops (for performance and memory). Masked_fill and some activations can also be made in-place
- kernel fusion (pare basti un decoratore e una funzione che usa solo roba cupy, stile jit). Non so se la differenza e' apprezzabile, prova prima a misurare le differenze sulla sigmoid
- group and sort the functions of Tensor's class because it's a mess to navigate


NOTE:
- Not _every_ operation has to be executed on GPU! See PermuteBackward to understand what I mean
---


# Optimizations
## Memory
- Implement **gradient checkpointing**, a technique that saves memory by partially computing on the go some operation of the forward pass instead of saving it;
- Implement **weight tying**: use the same embedding matrix for input and output
- **Change the optimizer**: Adam occupies a lot of space with his momentum vectors. Try Lion
- Implement **quantization**
- Implement an **occupied memory estimator**
- Implement **Mixed Precision Training (AMP)**
- Implement **RMSNorm** instead of LayerNorm;
- Set max_streams to 0 if you don't want to waste additional memory by worth_parallelizing()
- Precompute the dataset in a binary file (.npy) to keep in non-volatile memory and acces the data by index

## Performance
- <strike>Implement **QKV fusion** in the self attention blocks in order to compute them in parallel</strike>;
- Verify if your **CUDA stream infrastructure** is good;
- **Kernel fusion**! I already have written an idea above the forward of Sigmoid
- It could be useful to check some backward functions that breaks the contiguity of the gradient in order to restore it (for example look at TransposeBackward, MatmulBackward, PermuteBackward). I think that cupy manages things in background, but it's better to learn and to explictly ask for contiguity
- Implement **gradient accumulation**, a technique that simulates a much bigger batch size. mathematically is very similar too. It should be simple: update the params every $N$ batches, but first take the average of the accumulated gradients (so divide by $N$)


# Transformer
## Intelligence density
1) Large dataset
2) Quality dataset
3) Extensive training (even when the loss isn't diminishing anymore)
4) (to be verified) Suggested a percentage of math and coding to refine reasoning capabilities
5) Better tokenizer to compress the language representation, leaving more parameters for reasoning and comprehension

## Architecture
- Implement **RoPE**;
- Implement **SwiGLU**;
- Implement **RMSNorm** instead of LayerNorm. It will help with speed and memory;
- Implement MoE (if the memory allows... Which seems not);
- Implement Grouped Query Attention
- Implement **Multi Query Attention** (lightest version?)
- Implement Flash Attention (which should be the fastest)
- Implement decoder-only version

## Dataset
- Process Wikipedia's downloaded data

## Training
- You can start with a lower batch size to encourage the model to overcome sharp minimums (minimums that are deep but small, so they don't allow for good generalization) and gradually increasing it during training because the model gets smarter and needs a cleaner gradient
- Implement RLHF;
- Implement special tokens;
- Implement tool calls (it seems that it's implementet with a mix of special tokens and system prompt, but the model is trained to do so in one of the last phases of the training)
- Implement Test Time Compute

## Inference
- Build a RAG;
- Implement CoT (it's an inference-only thing? No training required at all?)

## Sanity checks
- All tensors are receiveng a gradient?
- All tensor have the desired dtype?

## Benchmarks & testing!
I bet that this phase will be interesting! Benchmarks are fundamental to understand the strong and weak points of the model in order to improve it in the next iteration. It's also useful (and interesting) to do a bit of experimentation on it to see which part of the network is responible of what. Look inside the model and try to interpet it!


---

# PeggleGameEnv
## Pre-requisiti
- CNN
- PPO
  
## Cose da fare
- Usare lo splendido codice delle maschere per crearsi una scan delle mappe di Peggle e usarle per automatizzarne la ricreazione sul mio gioco. Prima pero' le maschere devono codificare in qualche modo i pegs in movimento
  - => implementare motion dei pegs
  - => implementare pegs rettangolari di forma arbitraria
- logica cesto
- cercare di rendere simile la fisica di gioco. Visto che hai il codice che crea le maschere, non riesci a tracciare il moto della pallina? Prenderei una mappa che c'e' sia sull'originale che sul tuo, setterei il puntatore sulla stessa coordinata per entrambi, sparerei e salverei tutti i frame del tiro (penso che anche un paio al secondo vadano bene). Per ogni frame trovi le maschere, le unisci e le sottrai al frame originale. Dovrebbe quindi rimanerti solo la pallina nel frame originale. Da li' ricavane la traiettoria e vedi come si dovrebbero aggiustare i coefficienti nel tuo gioco per fare in modo che vengano rispettate.
Ogni parametro modificabile creaun grado di liberta', quindi, se non vuoi avere infiniti modi per descrivere quella singola traiettoria, raccogline un po' e trova i parametri che soddisfano tutte le traiettorie fatte in quel livello.
- La rete neurale, una CNN, prendera' in input uno stack di queste maschere. Stile RGB, ma stavolta ogni dimensione rappresenta un tipo di peg
- All'inizio trainala in modo civile in single player, anche su una mappa monca. Nella prima fase deve solo essere familiare con rimbalzi e cazzate varie. Come ha consigliato Gemini, niente rewards estremi: +1 per ogni arancione, +0.1 per ogni blu e -1 per i -25%. Rimani sul semplice e, soprattutto... Sul magnitudo **piccolo** o svacchi i gradienti! In ogni caso sara' da vedere meglio


## 

# Others

## Compatibiliy
- Check that the library can be installed and functions witouth CuPy imports
- Check that the setup.py works

## Visualization

## Testing
- Implement **seeds**. I am very curious to know how it works (Minecraft) and to implement a custom one, but I also have to implement one with PyTorch's algorithm or future comparisons will be harhs to do (like now);
- Implement a proper test infrastructure;



# Done
- <strike>Ho implementato .permute() e PermuteBackward per la fusione QKV, che sono **da testare**</strike>
- <strike>(gia' fatta) Ho implementato la fusione QKV nel caso della self attention, ma c'e' ancora da vedere per la fusione KV (cross attention). ALT! Non so se ha senso farla: io per ora voglio lavorare sugli LLM, che non hanno encoder, quindi e' possibile che non si</strike>
- <strike>Rifinirei il codice della fusione QKV (non mi piacciono le variabili dichiarate dentro, cosi' come i loro nomi monocarattere). In particolare sto self.head_dim lo metterei come attributo, sto sempre a ricalcolarlo oh</strike>
- <strike>Bisogna testare la backward del QKV fuso</strike>
- for the time being i won't implement inp-place ops. Tehy are funky, bizarre to implement and not recommended for the general use, so I won't bother for now
- <strike>Problema: la call register_backward_if_needed in contiguous non gestisce il caso in cui la grad_fn e' None perche' parte a cannone con grad_fn.parents() e da' errore. Ha senso eh, devo vedere se i parenti vogliono il gradiente, ma in questo caso che fare? Nella versione originale l'avevo gestita con un if grad_fn is not None, ma ti torna? Ha senso?</strike> Risolto con CloneBackward
- <strike>FONDAMENTALE: ridurre il tempo di istanza della classe Tensor. Fai di TUTTO per abbassare questa metrica e spaccherai tutto! E rendila piu' leggera se puoi</strike> <strike>non ho trovato bottleneck o altri miglioramenti da fare per ora. Bene!</strike>  E invece trovati! Benvenuto ._fast_init()!