# Dataset + Base + MMFF + Atom details nello stesso grafico

Il nuovo script `compare_three_representations.py` produce un confronto con
quattro serie affiancate. Il precedente script rimane invariato.

Con l'ambiente `d4-gen` attivo, dalla root del progetto:

```bash
# Riutilizza il report QM9 completo già calcolato
python compare_three_representations.py \
  --report outputs/distribution_comparison/qm9_three_representations/report.json \
  --output-dir outputs/joint_distribution_comparison/qm9_test
```

Questo comando legge il report, senza ricaricare le molecole o ricalcolare la
chimica. Confronta contemporaneamente il dataset e tutte e tre le
rappresentazioni, mantenendo le analisi separate dei seed.

Per ricalcolare l'analisi a partire dalle generazioni salvate:

```bash
python compare_three_representations.py \
  --reference-dataset qm9 --reference-split test

python compare_three_representations.py \
  --reference-dataset gdb13 --reference-split valid
```

Senza `--report`, si possono usare gli stessi parametri del precedente script:
`--generated NAME=PATH`, `--reference-root`, `--checkpoints-root`, `--model`,
`--max-reference`, `--max-generated`, ecc. La nuova analisi viene salvata nella
sottocartella `analysis/` della nuova cartella dei risultati. I dataset e i
checkpoint sono letti; le precedenti analisi restano invariate.

Con `--report`, riferimento e split sono quelli del report: per cambiarli si
lancia una nuova analisi omettendo quel parametro.

I risultati comprendono:

- `overview.png` e `overview.pdf`: sei pannelli con tipi MMFF, cariche formali,
  ibridazione, appartenenza agli anelli, aromaticità e tipi di legame;
- un PNG e un PDF per descrittore: in alto le quattro distribuzioni affiancate,
  in basso gli scarti delle tre generazioni rispetto al dataset;
- `comparison.csv`: le frequenze rappresentate, media, deviazione standard,
  numero di seed e scarti dal dataset;
- `joint_comparison.json`: tabelle, seed, avvisi, provenienza e hash del report
  originale.

Le frequenze nei grafici sono percentuali. Lo scarto è in **punti percentuali**:
se il dataset ha il 20% e una generazione il 25%, lo scarto è +5 punti.
Le medie danno lo stesso peso a ogni seed disponibile. Le barre di errore sono
le deviazioni standard campionarie tra seed (`n-1`), non intervalli di confidenza;
con un solo seed la deviazione standard è indefinita e la barra non viene
mostrata. Il dataset è un unico riferimento empirico, senza barre tra seed.

I seed mancanti sono segnalati. Una rappresentazione interamente assente viene
indicata in legenda come non disponibile, senza inventare frequenze nulle.
Le frequenze atomiche contano atomi, quelle dei legami contano legami; carica
molecolare e presenza di gruppi contano molecole valide. Per i profili chimici
valgono i criteri del report originale: frammento maggiore sanitizzato e
copertura MMFF esplicita.

`--top-k 20` limita le categorie dei grafici individuali; le altre vengono
raggruppate in "Altre categorie". Nella panoramica il limite è 10, per leggibilità.
Le presenze di gruppi funzionali sono frazioni indipendenti e non vengono
sommate in una categoria residuale. CSV e JSON del nuovo script descrivono le
categorie rappresentate; il report originale conserva tutte quelle individuali.
`--no-pdf` produce solo i PNG.
