Verifica del 16 ottobre 2026 sul branch `drigoni/atom_types2`, commit di partenza
`f7784c6`. Le correzioni descritte sono nel working tree.

Il controllo finale passa per tutti i dataset presenti: 20 cache attive,
1.618.900 record sorgente contando separatamente le tre varianti QM9,
3.237.800 record nelle cache complete e nei rispettivi split. I risultati
strutturati sono in `outputs/dataset_audit/final.json`.

| Dataset | Completo | Training | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| QM9 | 130.194 | 110.194 | 10.000 | 10.000 |
| QM9 MMFF | 130.194 | 110.194 | 10.000 | 10.000 |
| QM9 atom-details | 130.194 | 110.194 | 10.000 | 10.000 |
| GDB13 | 978.896 | 783.118 | 97.889 | 97.889 |
| ZINC | 249.422 | 188.611 | 35.925 | 24.886 |

Per ogni cache sono stati controllati indici e confini dei grafi, categorie e
one-hot, valori finiti, coordinate centrate, legami simmetrici, assenza di
self-loop e archi duplicati, istogrammi e marginals ricalcolati dai tensori.
Ogni split è stato confrontato, attributo per attributo, con i corrispondenti
record della cache completa. ZINC conserva il test predefinito; gli altri split
seguono le configurazioni del progetto con seed 0.

Per **ogni molecola** dei cinque dataset sono stati confrontati elementi,
cariche, legami, coordinate e target con i sorgenti RDKit salvati in `raw/`.
Per QM9 MMFF sono stati ricalcolati tutti i tipi MMFF; per QM9 atom-details sono
stati ricalcolati appartenenza agli anelli, aromaticità e ibridazione. Nessuna
discrepanza rilevata. Questa verifica certifica la coerenza con i sorgenti locali,
non una nuova validazione delle misure chimiche originarie.

Le tre varianti QM9 hanno gli stessi elementi e gli stessi record nello stesso
ordine: 4 categorie di elemento, 3 di carica, 39 tipi MMFF oppure categorie
atom-details di dimensioni 2/2/3. Le tre categorie dei legami nei grafi sparsi e le
quattro componenti delle relative marginals sono corrette: la classe aggiuntiva
rappresenta l'assenza di legame nella rappresentazione densa.

Sono stati trovati e corretti due problemi nelle cache:

- `qm9` e `qm9_mmff` avevano training di 104.155 record, un vecchio `val` di
  26.038 record, un test di un solo record e un `valid` di 10.000 record quasi
  interamente sovrapposto al training. I default chiedevano 80% + 20% + 20% e il
  generatore di split troncava gli indici eccedenti. Lo script di confronto
  poteva attivare questi default. Sono stati ripristinati gli split previsti
  dalle configurazioni QM9, già presenti correttamente in atom-details.
- Su GDB13 e ZINC le marginals dei legami di completo e training risentivano
  dell'accumulo dei conteggi in float32. La discrepanza massima osservata sul
  completo era circa 0,000115 e 0,0000125 rispettivamente. Sono state ricalcolate
  le statistiche e l'accumulo futuro usa float64; i grafi sono rimasti invariati.

Il codice ora rifiuta split sovradimensionati prima di scrivere file e verifica
le dimensioni degli split QM9/GDB13 esistenti prima del training. I default QM9
coincidono con i preset. Il confronto delle distribuzioni carica la partizione
esistente senza crearne una nuova e rifiuta una cache di riferimento incompleta.
Tre esecuzioni del confronto sulle varianti QM9 hanno confermato che nessun file
dei dataset veniva modificato.

Le partizioni precedenti e gli indici usati per la riparazione sono conservati in:

- `datasets/qm9/split_backups/audit_1792143833635256850/`
- `datasets/qm9_mmff/split_backups/audit_1792143838923126460/`

Le statistiche precedenti sono conservate in:

- `datasets/gdb13/stats_backups/audit_1792144052011165454/`
- `datasets/zinc/stats_backups/audit_1792144336438202427/`

Sono state corrette anche le due anomalie dei modelli:

- D4 saltava la conversione one-hot di tutte le variabili con una sola categoria.
  Una variabile costante deve comunque diventare un tensore `[batch, nodi, 1]`
  di uno; lasciarla come indici zero produce dimensioni errate. La condizione
  distingue ora la distanza continua dalle variabili categoriche. Le categorie
  dei dataset attuali hanno tutte almeno due valori, quindi questa anomalia
  non aveva corrotto le cache esaminate.
- MiDi mediava le KL dei diversi stati ausiliari nella stima NLL. Nella
  distribuzione congiunta quei termini si sommano: per esempio 0,2 + 0,3 + 0,4
  dà 0,9, mentre il codice precedente conteggiava 0,3. Per atom-details il
  contributo ausiliario era quindi diviso per tre; MMFF ha un solo stato ed è
  invariato. Sono stati corretti la stima e i log aggregati. La media nella loss
  di training rimane una scelta distinta per bilanciare i diversi obiettivi.

I nove test di regressione passano. I quattro preset D4/MiDi con MMFF/atom-details
hanno inoltre passato forward, backward con gradienti finiti e un passo inverso
di generazione su batch reali, su CPU e senza compilazione. Queste prove non
sostituiscono un training completo o una verifica della compilazione su GPU.

Rimane una proprietà dei sorgenti da considerare negli esperimenti: lo split è
per record, non per SMILES canonico. Ci sono quindi SMILES condivisi fra record
di split diversi, pur con indici disgiunti:

| Dataset | Training/validation | Training/test | Validation/test |
| --- | ---: | ---: | ---: |
| Ogni variante QM9 | 20 | 24 | 1 |
| GDB13 | 29 | 27 | 0 |
| ZINC | 366 | 257 | 42 |

Una separazione rigorosa per identità molecolare richiederebbe una diversa
politica di split per gruppi. La riparazione conserva la politica dei preset.
Risultati ottenuti usando le vecchie partizioni incoerenti devono essere
riesaminati; se quei record sono stati usati nel training, cambiare il test
non elimina la sovrapposizione preesistente. Le NLL MiDi atom-details precedenti
non usavano la stessa formula corretta. Le varianti estese di GDB13/ZINC non sono
presenti localmente e non sono state certificate da questa verifica.

Per ripetere l'audit senza modificare dati:

```bash
conda activate d4-gen
python audit_preprocessed_datasets.py --other-chemistry-samples 1000000
python -m unittest tests.test_dataset_integrity -v
```

Le opzioni `--repair-qm9-splits` e `--repair-statistics` sono esplicite e conservano
backup. Senza tali opzioni l'audit legge soltanto i dataset e scrive il report.
