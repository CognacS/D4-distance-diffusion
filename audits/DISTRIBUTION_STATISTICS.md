# Test statistici tra rappresentazioni

Il nuovo script `compare_distribution_statistics.py` legge i `metrics.json`
prodotti da `score_generated_distributions.py`. Confronta i modelli sullo stesso
riferimento, usando i seed indipendenti di training come repliche.

Dalla root del repository, nell'ambiente `d4-gen`:

```bash
# Un solo riferimento
python compare_distribution_statistics.py \
  --scores outputs/distribution_comparison/qm9_test_job1930870/scores/metrics.json \
  --output-dir outputs/distribution_comparison/qm9_test_job1930870/statistics

# Train e test analizzati separatamente, con correzione multipla comune
python compare_distribution_statistics.py \
  --scores \
    train=outputs/distribution_comparison/qm9_train_job1930862/scores/metrics.json \
    test=outputs/distribution_comparison/qm9_test_job1930870/scores/metrics.json \
  --output-dir outputs/distribution_statistics/qm9_train_test
```

Il file Slurm `cluster/experiments/compare_distributions.slurm` ora esegue
nell'ordine analisi, score con Wasserstein dedicata e test statistici. Lo split
rimane quello selezionato dalla variabile `reference_split`.

## Quale domanda viene testata

L'ipotesi nulla è che le distribuzioni dei punteggi dei seed siano scambiabili
tra modelli, condizionatamente al riferimento fissato. Si assumono repliche di
training indipendenti. Avere seed chiamati 0, 1 e 2 in tutte le rappresentazioni
non dimostra da solo un appaiamento sperimentale. Se esiste un vero disegno
appaiato, serve un'analisi coerente con quel disegno.

Per ciascun riferimento vengono calcolati:

- un test globale tra i modelli, con statistica F di ANOVA e distribuzione nulla
  ottenuta tramite permutazioni, senza usare l'approssimazione parametrica F;
- confronti bilaterali a coppie, con statistica `abs(media_A - media_B)`;
- differenze medie e delta di Cliff descrittivo: un delta positivo indica che A
  ha distanze inferiori più spesso di B.

Le unità del test sono i **seed**, non gli atomi, i descrittori o le 10.000
molecole generate da ogni modello. La validità statistica del test richiede
l'assunzione di scambiabilità sotto H0.
[Documentazione delle permutazioni indipendenti](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

## Limite dei tre seed

Con 3 seed contro 3 ci sono `binomiale(6,3) = 20` assegnazioni. Per il test
bilaterale a coppie, l'assegnazione osservata e quella con i gruppi scambiati
sono entrambe altrettanto estreme. Il p-value minimo esatto è quindi **2/20 =
0.10**, prima di qualsiasi correzione multipla.

Questi confronti a coppie non possono dunque raggiungere alpha 0.05, anche
quando i tre punteggi di un modello sono tutti inferiori a quelli dell'altro.
Un p-value non significativo non dimostra equivalenza: con così poche repliche
la potenza è limitata. Servono più seed indipendenti per una valutazione più
informativa della variabilità di training; aumentare solo le molecole generate
non aumenta il numero di repliche di training.

Il test globale su tre modelli ha 1.680 assegnazioni quando tutti hanno tre
seed e può avere una risoluzione diversa. Un eventuale rifiuto globale non
identifica automaticamente quale coppia differisce.

## Più modelli o più molecole?

Più allenamenti indipendenti per ciascuna rappresentazione aumentano le repliche
per confrontare le rappresentazioni tenendo conto della variabilità di training.
Più molecole dallo stesso checkpoint riducono invece il rumore della stima
delle distribuzioni, mantenendo fisso il modello allenato.

Una generazione da 50.000 molecole e cinque generazioni indipendenti da 10.000
possono stimare la stessa distribuzione del medesimo checkpoint. Per ottenere
50.000 campioni indipendenti servono stati/semi casuali di generazione diversi,
senza rigenerare deterministicamente le stesse molecole. I campioni si possono
riunire per calcolare una distribuzione da 50.000 osservazioni; per il test tra
modelli resta comunque una sola replica di training.

Tenere distinti seed di training e seed di generazione. Cinque set generati
dallo stesso checkpoint non devono essere registrati come cinque allenamenti.
Per confronti equi usare lo stesso numero di campioni per checkpoint tra le
rappresentazioni e lo stesso riferimento. Cinque generazioni da 10.000 possono
anche servire a stimare la variabilità della sola fase di campionamento.

L'analizzatore attuale seleziona un solo file di generazione per seed di
training: in presenza di più file sceglie quello più recente e lo segnala.
Non aggrega automaticamente cinque file da 10.000. Per analizzare 50.000
campioni insieme occorre fornirli in un unico file compatibile o aggiungere
una modalità esplicita di aggregazione, senza aumentare artificialmente il
numero di seed di training.

## Confronti multipli

La correzione Holm usa due famiglie dichiarate:

1. **Primaria:** test globale e confronti a coppie della TVD complessiva, su
   tutti i riferimenti passati alla stessa esecuzione.
2. **Esplorativa:** Jensen–Shannon complessiva, cinque famiglie TVD e Wasserstein
   per descrittore numerico, inclusi i rispettivi test globali e a coppie.

I p-value grezzi e corretti sono entrambi conservati. Eseguire train e test
separatamente corregge i confronti entro ogni esecuzione; passarli insieme
include entrambi i riferimenti nella stessa famiglia di correzione.
[Correzione Holm](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).

Le permutazioni sono esatte fino al limite `--max-permutations` (default 20.000).
Oltre quel numero si usa Monte Carlo riproducibile con `--seed`, applicando la
correzione `(estremi+1)/(simulazioni+1)` per non produrre p-value nulli.

## File prodotti e interpretazione

- `summary.md`: confronti primari e limiti interpretativi;
- `statistics.json`: test, seed usati, numerosità, metodo esatto/Monte Carlo,
  fonti e hash dei file di input;
- `statistical_tests.csv`: risultati primari ed esplorativi;
- `primary_pvalues.png/.pdf`: p-value della TVD complessiva, prima e dopo Holm.

Il programma non effettua un test diretto train-vs-test. Quelle distanze usano
riferimenti diversi, con numerosità diverse (110.194 contro 10.000 per QM9):
anche il rumore delle distanze empiriche cambia. Non vanno interpretate come
osservazioni identicamente calibrate. Non viene ricalcolata l'incertezza delle
singole molecole del riferimento.

## Dove trovare Wasserstein

Era già nella colonna `wasserstein` di `scores/distances_by_seed.csv` e nel
JSON per ciascun descrittore numerico. Non era visibile in `group_scores.csv`,
che contiene le cinque famiglie della TVD.

Ora ogni esecuzione degli score produce anche:

- una sezione Wasserstein nel `scores/summary.md`, con tabella di media, SD,
  numero di seed di training, unità e minimo medio per descrittore;
- `scores/wasserstein_summary.csv`: media, deviazione standard, numero di seed
  e unità per ogni descrittore e rappresentazione;
- `scores/wasserstein_by_seed.csv`: valori individuali;
- `scores/wasserstein.png` e `.pdf`: pannelli con media, deviazione standard e
  singoli seed, usando scale separate.

Wasserstein viene usata su cariche formali e quantità numeriche, come numero
di atomi, legami o anelli. Gli ID dei tipi MMFF sono etichette categoriche:
la differenza numerica tra due ID non rappresenta una distanza chimica. Per
quelle categorie rimangono appropriate TVD e Jensen–Shannon. Non si costruisce
una somma Wasserstein tra descrittori con unità diverse.

Verifica automatica:

```bash
python -m unittest tests.test_distribution_statistics tests.test_distribution_scores -v
```
