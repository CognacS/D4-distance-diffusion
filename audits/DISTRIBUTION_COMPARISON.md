# Confrontare la chimica generata con QM9

Eseguire dalla root del repository, con l'ambiente `d4-gen` attivo:

```bash
python compare_generated_distributions.py
```

Il comando confronta D4 base, MMFF e atom_details con **lo stesso test set QM9**.
Cerca le cartelle `qm9-d4`, `qm9-d4-mmff-v2` e `qm9-d4-atom-details-v2` in
`checkpoints/`; per le rappresentazioni estese usa le cartelle senza `-v2` se
quelle preferite non contengono generazioni. Se trova più cartelle compatibili,
stampa quella scelta e quelle ignorate. Il suffisso della cartella è solo una
convenzione di ricerca: i vocabolari vengono letti dai dataset, le proprietà
vengono ricalcolate dalle strutture e il seed viene letto dai metadati.

## Scegliere il riferimento o le generazioni

```bash
# Training set o validation set, senza ricreare gli split
python compare_generated_distributions.py --reference-split train
python compare_generated_distributions.py --reference-split valid

# Una sola rappresentazione, con tutti i suoi seed disponibili
python compare_generated_distributions.py --representations mmff

# Cartelle esplicite, anche fuori dalla struttura standard
python compare_generated_distributions.py \
  --generated base=checkpoints/qm9-d4 \
  --generated mmff=checkpoints/qm9-d4-mmff-v2 \
  --generated atom_details=checkpoints/qm9-d4-atom-details-v2 \
  --reference-split test \
  --output outputs/qm9_comparison/report.json

# Prova con un sottoinsieme riproducibile: il limite generato è PER seed
python compare_generated_distributions.py --max-reference 100 --max-generated 100
```

`--reference-dataset` è `qm9` per default; `--reference-root` permette di usare
un'altra directory del dataset. Sono accettati anche `validation` e `val` come
alias di `valid`, e `full` per l'intero dataset. `--model midi` cambia la ricerca
automatica delle cartelle. I file dei grafi vengono letti interamente anche se
si limita il numero di molecole da analizzare.

## Seed e provenienza

I seed attesi sono 0, 1 e 2. Ogni seed viene identificato tramite `config.yaml`,
`.hydra/config.yaml` nella run oppure i file config/metadati salvati in
`checkpoints/wandb/run-…/files/`. Il contatore `v0`, `v1`, ecc. non identifica il
seed. Conservare i metadati W&B quando si copiano le generazioni su un'altra
macchina. Con `--checkpoints-root` si indica una diversa root dei checkpoint.

I seed mancanti o non identificabili sono segnalati in console e nel report.
Una rappresentazione senza seed disponibili rimane indicata come mancante; se
non si trova nessuna run utilizzabile il comando termina con errore.
`--expected-seeds` permette di specificare un insieme diverso.

Si usa un solo file per seed. Se una run contiene più generazioni si sceglie il
file modificato più recentemente. Se ci sono più run per lo stesso seed si
sceglie quella con la generazione più recente. Entrambe le scelte producono un
avviso con il file selezionato; percorsi espliciti permettono di scegliere una
run o uno specifico file.

## Cosa viene misurato

RDKit ricostruisce la molecola usando elementi, cariche formali e legami. Gli
attributi ausiliari predetti non vengono usati per ricostruirla. I vocabolari
sono letti da `raw/stats.json` del dataset di ciascuna rappresentazione; i
preprocessati di riferimento sono letti direttamente, senza preparare dataset,
scaricare dati o rigenerare partizioni.

**Tutte e tre le rappresentazioni ricevono gli stessi descrittori:** elementi,
cariche formali atomiche, carica totale della molecola, tipi MMFF, aromaticità,
appartenenza agli anelli, ibridazione, ambienti atomici, legami, anelli e gruppi
funzionali SMARTS. Si confrontano anche elemento–carica, MMFF–carica ed
elemento–caratteristiche atom_details. I tipi MMFF sono ricalcolati con idrogeni
aggiunti temporaneamente; vengono conteggiati solo gli atomi della molecola
analizzata, senza gli idrogeni aggiunti.

Le proprietà chimiche riguardano il frammento maggiore sanitizzato delle
molecole valide. La carica molecolare è quindi la carica di quel frammento.
Validità, componenti connesse e statistiche del grafo originale restano
esplicite. Le molecole ripetute contribuiscono ripetutamente alle frequenze;
l'unicità è riportata separatamente.

Il report specifica i denominatori: le frequenze atomiche contano atomi,
quelle dei legami contano legami e quelle degli anelli contano anelli. Molecole
più grandi contribuiscono quindi più atomi e legami. Carica molecolare e
presenza di un gruppo funzionale contano molecole valide. I gruppi SMARTS
possono sovrapporsi. La frequenza delle occorrenze è distinta dalla frazione di
molecole che contengono un gruppo.

MMFF può non essere disponibile per qualche molecola valida: il report indica
quante molecole e quanti atomi partecipano a quel confronto. Una distribuzione
non calcolabile è indefinita, non una distribuzione con probabilità zero.

## Risultati e interpretazione

Per default i risultati sono in
`outputs/distribution_comparison/<dataset>_<model>_<split>_<data>/`:

- `report.json`: file selezionati, seed, avvisi, conteggi, copertura MMFF,
  distribuzioni per seed e differenze rispetto al riferimento;
- grafici PNG: riferimento e tre rappresentazioni affiancati, con media per
  seed e barre di deviazione standard.

Le medie assegnano lo stesso peso a ogni seed disponibile, anche se il numero
di molecole valide differisce. La deviazione standard è campionaria (`n-1`);
con un solo seed è `null`. Il campo `n` indica quanti seed contribuiscono a ogni
stima. Le categorie assenti in una distribuzione valida contano come zero;
le distribuzioni interamente non calcolabili sono escluse dalla media.

La **TVD**, tra 0 e 1, misura le differenze nelle frequenze categoriche. La
**distanza di Wasserstein** misura le differenze in numero di atomi, legami,
anelli e componenti. Zero indica distribuzioni coincidenti per quel descrittore.
Sono confronti descrittivi, non test di significatività o prove che le intere
distribuzioni molecolari coincidano.

`novelty_vs_reference` confronta gli SMILES con lo split selezionato. Con il test
set non coincide con la novità rispetto al training riportata nei log.
I grafici mostrano le categorie più frequenti (`--plot-top-k`, default 20);
il JSON conserva tutte le categorie. Le categorie omesse sono raccolte in
`Other`, eccetto le presenze di gruppi funzionali, che sono frazioni indipendenti
e non si possono sommare come una distribuzione categorica.

Verifica dei calcoli e della selezione dei seed:

```bash
python -m unittest tests.test_generated_distributions -v
```
