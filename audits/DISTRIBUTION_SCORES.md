# Misurare la somiglianza delle distribuzioni generate

`score_generated_distributions.py` legge il report multi-seed già prodotto da
`compare_generated_distributions.py`, calcola distanze numeriche e genera una
graduatoria, grafici, tabelle CSV e un riepilogo leggibile. Non ricalcola le
molecole e non modifica i precedenti script o report.

Con l'ambiente `d4-gen` attivo:

```bash
# Default: report QM9 con riferimento test già calcolato
python score_generated_distributions.py

# Report esplicito, anche relativo a un altro dataset o split
python score_generated_distributions.py \
  --report outputs/distribution_comparison/qm9_three_representations/report.json \
  --output-dir outputs/distribution_scores/qm9_test
```

Il riferimento è quello registrato nel report. Per confrontare train, valid o
un altro dataset, generare prima il rispettivo report e passarlo con `--report`.
I risultati QM9 verificati sono in `outputs/distribution_scores/qm9_test/`.

## Metriche

La **TVD** è la metrica principale per confrontare le frequenze:

`TVD(p,q) = 0.5 * somma_categorie |p - q|`

È compresa tra 0 e 1; valori più bassi indicano distribuzioni più simili.
Una categoria presente solo da una parte viene inclusa nel calcolo, con
probabilità zero dall'altra. Si usano tutte le categorie del report, senza
limiti top-k o raggruppamenti grafici.
[Definizione e interpretazione della TVD](https://www.stat.berkeley.edu/~aldous/RWG/Book_Ralph/Ch2.S4.html).

La **distanza Jensen–Shannon** viene riportata come confronto secondario. Si
usa la radice della divergenza, con logaritmi base 2, quindi tra 0 e 1.
[Definizione SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html).

La **Wasserstein-1** viene calcolata per descrittori numerici, per esempio numero
di atomi, legami e anelli. Tiene conto anche della distanza fra i valori:
spostare probabilità da 5 a 6 atomi pesa meno che spostarla da 5 a 9.
Rimane nelle unità originali e non viene sommata alle metriche categoriche.
[Definizione SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).

Le presenze di gruppi funzionali non sono categorie esclusive: per ciascun
gruppo si confronta una distribuzione Bernoulli. La sua TVD è `|p - q|`; si
media poi sui gruppi. Normalizzare insieme tutte le presenze sarebbe scorretto.

## Indice complessivo dichiarato

Per ottenere una graduatoria leggibile si usa una **media gerarchica delle
TVD**, costruita per questa analisi. Non è una metrica molecolare standard.
Ogni famiglia ha di default peso 20%; le proprietà dentro una famiglia hanno
lo stesso peso:

| Famiglia | Proprietà |
|---|---|
| Composizione e cariche | elementi, cariche atomiche, carica molecolare |
| Ambienti atomici | tipi MMFF, ibridazione, appartenenza agli anelli, aromaticità |
| Legami | tipi di legame |
| Struttura molecolare | distribuzioni del numero di atomi, legami, anelli e componenti delle molecole valide |
| Gruppi funzionali | presenza media dei gruppi SMARTS |

Prima si calcolano gli indici per ciascun seed; poi media e deviazione standard
campionaria (`n-1`) tra i seed disponibili. In questo modo la variabilità del
punteggio tiene conto della covariazione tra le proprietà dello stesso seed.
Ogni seed pesa uguale, anche se il numero di molecole valide differisce.

I tipi MMFF e atom_details sono ricalcolati dal precedente analizzatore per
**tutte** le rappresentazioni, compresa quella base. Il punteggio usa quindi
lo stesso insieme di proprietà per tutti i modelli.

Si può dichiarare un'altra pesatura. Per esempio, dare doppio peso agli
ambienti atomici, lasciando le altre famiglie a peso 1:

```bash
python score_generated_distributions.py --group-weights atomic_environments=2
```

Sono accettati anche `composition`, `bonds`, `molecular_structure` e
`functional_groups`. Un peso zero esclude la famiglia. I pesi vengono
normalizzati e salvati nel report. È consigliabile scegliere i pesi in base
alla domanda scientifica, prima di osservare quale modello risulti primo.

Il report include anche la graduatoria ottenuta eliminando una famiglia alla
volta, mantenendo gli stessi seed con punteggio completo: mostra quanto
l'ordinamento dipenda dalla scelta dei descrittori.

## File prodotti

- `summary.md`: graduatoria TVD/Jensen–Shannon e tabella Wasserstein per tutti i
  descrittori numerici disponibili, con media, SD, numero di seed, unità e modello
  con il minimo medio; definizioni, pesi, sensibilità e limiti;
- `metrics.json`: distanze per seed e descrittore, riepiloghi, qualità e copertura;
- `ranking.csv`: graduatoria complessiva;
- `group_scores.csv`: distanze per famiglia e relativi pesi;
- `distances_by_seed.csv`: TVD, Jensen–Shannon e Wasserstein per descrittore e seed;
- `wasserstein_summary.csv` e `wasserstein_by_seed.csv`: Wasserstein dedicata, con medie/SD e valori individuali;
- `wasserstein.png/.pdf`: distanze Wasserstein in pannelli con scale e unità separate;
- `ranking.png/.pdf`: indici complessivi, valori dei seed e deviazioni standard;
- `groups_tvd.png/.pdf`: confronto delle cinque famiglie;
- `descriptors_tvd.png/.pdf`: confronto di tutte le singole proprietà.

Nelle mappe un bordo verde indica il minimo medio della riga: non implica una
differenza statisticamente significativa.

## Interpretazione

Il punteggio confronta le **distribuzioni marginali dei descrittori scelti**.
Non certifica che le distribuzioni delle strutture molecolari complete siano
uguali. Un valore 0.01 non significa che il 99% delle molecole sia corretto.

Le distanze empiriche dipendono anche dalle dimensioni dei campioni. I conteggi
per seed sono conservati nel JSON. Per il confronto QM9 disponibile, ogni seed
contiene 10.000 generazioni e il riferimento contiene 10.000 molecole.

Le proprietà chimiche riguardano i frammenti maggiori sanitizzati validi.
Validità e copertura MMFF sono riportate separatamente: una distribuzione
simile tra i campioni validi non garantisce un buon tasso di validità.

I seed mancanti vengono segnalati. Un descrittore indefinito non vale zero:
se manca una proprietà richiesta da una famiglia attiva, il seed non riceve
un indice complessivo. Con un solo seed la deviazione standard è indefinita.
Le barre mostrano la variabilità tra seed, non intervalli di confidenza.
Non si assume che seed con lo stesso numero siano misure statisticamente
appaiate. Lo script dei soli score non calcola p-value: per i test di permutazione
usare [compare_distribution_statistics.py e la relativa guida](DISTRIBUTION_STATISTICS.md).

Verifiche numeriche:

```bash
python -m unittest tests.test_distribution_scores -v
```
