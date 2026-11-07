# GridPilot AI – Documentația modelului de estimare a ampacității

> **Scop:** documentarea modelului ML care estimează capacitatea de transport a curentului electric (ampacitatea) unui conductor, pe baza condițiilor meteorologice și a caracteristicilor segmentului de linie.
>
> **Important:** țintele folosite la antrenare și evaluare sunt generate prin simulare cu LineRate, folosind modelul fizic CIGRE Technical Brochure 601. Ele nu sunt măsurători reale dintr-o rețea electrică.

## 1. Prezentare generală

Modelul GridPilot AI estimează ampacitatea unui segment de linie electrică în amperi (A). În loc să returneze o singură valoare, sistemul produce trei estimări:

- **Q05** – limita inferioară a intervalului predictiv;
- **Q50** – estimarea mediană, folosită ca predicție centrală;
- **Q95** – limita superioară a intervalului predictiv.

Intervalul Q05–Q95 este conceput pentru a descrie incertitudinea aproximării ML față de țintele fizice simulate. Nu reprezintă un interval complet al incertitudinii din exploatarea unei rețele reale.

## 2. Sursa și natura datelor

Setul de date folosit pentru antrenarea finală conține **105.192 de observații** și **16 coloane**. Fiecare observație reprezintă o combinație între un moment temporal și un segment de linie.

Ținta de predicție este:

- `target_ampacity_a` – ampacitatea calculată de motorul fizic LineRate, exprimată în amperi.

Etichetele au fost generate cu `LineRate.Cigre601`, conform CIGRE TB 601, pentru un conductor simulat. Prin urmare, modelul învață să aproximeze rezultatele acestui model fizic pe baza caracteristicilor de intrare.

### 2.1. Variabile de intrare

Modelul folosește 12 caracteristici:

| Caracteristică | Descriere |
|---|---|
| `air_temp_c` | Temperatura aerului, °C |
| `wind_speed_ms` | Viteza vântului, m/s |
| `wind_angle_of_attack_deg` | Unghiul de atac al vântului, grade |
| `crosswind_ms` | Componenta transversală a vântului, m/s |
| `parallel_wind_ms` | Componenta paralelă a vântului, m/s |
| `solar_radiation` | Radiația solară |
| `sin_solar_altitude` | Sinusul unghiului de elevație solară |
| `hour_sin` | Componentă ciclică pentru ora zilei |
| `hour_cos` | Componentă ciclică pentru ora zilei |
| `doy_sin` | Componentă ciclică pentru ziua anului |
| `doy_cos` | Componentă ciclică pentru ziua anului |
| `line_azimuth_deg` | Azimutul segmentului de linie, grade |

În plus, setul de date păstrează `timestamp_utc`, `segment_id` și `data_quality`, folosite la organizarea, validarea și identificarea observațiilor. Aceste coloane nu fac parte din lista celor 12 caracteristici transmise direct estimatorilor.

### 2.2. Curățarea și construirea setului de date

Pipeline-ul de construire a datelor:

1. validează observațiile meteorologice;
2. elimină rândurile cu date meteorologice invalide;
3. elimină rândurile care referă segmente necunoscute;
4. calculează etichetele fizice cu LineRate/CIGRE TB 601;
5. exclude calculele fizice eșuate;
6. construiește caracteristicile meteorologice, temporale și geometrice;
7. păstrează observațiile valide și le sortează după timestamp și segment.

Funcția relevantă din proiect este `build_training_dataset(...)`, iar validarea setului de date preconstruit este realizată de `_check_dataset(...)`.

## 3. Împărțirea datelor

Evaluarea finală folosește o împărțire cronologică, nu o împărțire aleatorie. Astfel, modelul este testat pe perioade ulterioare celor folosite la antrenare.

| Set | Număr observații | Interval temporal (UTC) | Proporție aproximativă |
|---|---:|---|---:|
| Train | 52.596 | 2022-01-01 00:00 – 2024-01-01 11:00 | 50% |
| Validation | 26.226 | 2024-01-02 12:00 – 2024-12-31 17:00 | 25% |
| Test | 26.226 | 2025-01-01 18:00 – 2025-12-31 23:00 | 25% |
| **Total** | **105.048** | | |

Între perioade este aplicată o pauză temporală de **24 de ore** (`gap_hours=24`), pentru a reduce riscul ca observațiile apropiate temporal să se distribuie imediat de o parte și de alta a limitelor split-ului.

Observație: setul inițial construit conține 105.192 de rânduri, în timp ce cele trei subseturi ale split-ului final însumează 105.048 de rânduri. Diferența de 144 de observații este compatibilă cu excluderea intervalelor de pauză dintre subseturi.

## 4. Arhitectura și antrenarea modelului

Modelul principal folosește estimatori **HistGradientBoostingRegressor** din scikit-learn, cu obiective de regresie cuantila pentru estimarea Q05, Q50 și Q95.

Sunt antrenate modele separate pentru cuantile, iar la inferență rezultatele sunt combinate și calibrate.

### 4.1. Configurația de antrenare

Configurația salvată în metadatele modelului:

| Parametru | Valoare |
|---|---:|
| Algoritm | HistGradientBoostingRegressor |
| `random_state` | 42 |
| `train_frac` | 0,50 |
| `val_frac` | 0,25 |
| `gap_hours` | 24 |
| `max_iter` | 300 |
| `learning_rate` | 0,08 |
| `max_leaf_nodes` | 31 |
| `min_samples_leaf` | 40 |
| `l2_regularization` | 1,0 |
| `backtest_folds` | 4 |
| `weight_moderate` | 3 |
| `weight_severe` | 5 |
| `moderate_temp` | 28 °C |
| `moderate_wind` | 2 m/s |
| `severe_temp` | 30 °C |
| `severe_wind` | 1 m/s |
| `calibrate_by_regime` | True |
| `monotonic` | False |

Ponderile mai mari sunt folosite pentru observațiile care intră în regimurile meteorologice moderate și severe definite de pragurile de mai sus.

### 4.2. Modelele salvate în bundle

Bundle-ul final conține următoarele componente:

- `hgb_q05` – estimatorul cuantilei inferioare;
- `hgb_q50` – estimatorul cuantilei mediane;
- `hgb_q95` – estimatorul cuantilei superioare;
- `q05_shift_a` – deplasarea globală de calibrare Q05;
- `q05_shift_day_a` – deplasarea Q05 pentru condiții de zi;
- `q05_shift_night_a` – deplasarea Q05 pentru condiții de noapte.

Deplasările de calibrare înregistrate în bundle:

| Regim | Deplasare Q05 |
|---|---:|
| Zi | 3,3617 A |
| Noapte | 0,1504 A |

În `predict_quantiles(...)`, regimul zi/noapte este determinat cu `sin_solar_altitude > 0`. Deplasarea corespunzătoare este scăzută din predicția Q05.

Dacă estimările cuantilelor se intersectează (`Q05 > Q50` sau `Q50 > Q95`), implementarea sortează cele trei valori pentru fiecare observație, astfel încât ordinea finală să fie Q05 ≤ Q50 ≤ Q95. Valorile negative sunt limitate la zero.

## 5. Performanța modelului

Performanța de mai jos este calculată cu logica de inferență folosită la deployment, inclusiv calibrarea Q05 și repararea crossing-ului cuantilelor.

### 5.1. Rezultate Train / Validation / Test

| Metrică | Train | Validation | Test |
|---|---:|---:|---:|
| Număr observații | 52.596 | 26.226 | 26.226 |
| MAE (A) | 4,0127 | 4,9316 | 4,8306 |
| RMSE (A) | 5,9778 | 7,7296 | 7,2691 |
| R² | 0,9991 | 0,9984 | 0,9986 |
| Q05 coverage (%) | 98,8478 | 96,7742 | 97,0564 |
| Interval coverage Q05–Q95 (%) | 95,9560 | 92,1757 | 92,1986 |
| Lățime medie interval (A) | 39,2209 | 39,4710 | 39,2658 |
| Quantile crossing înainte de reparare (%) | 23,0664 | 26,8703 | 26,4547 |

### 5.2. Interpretarea metricilor

- **MAE (Mean Absolute Error):** diferența absolută medie dintre Q50 și ținta LineRate. Pe setul TEST, eroarea medie este de aproximativ **4,83 A**.
- **RMSE (Root Mean Squared Error):** penalizează mai mult erorile mari. Pe TEST este de aproximativ **7,27 A**.
- **R² (coeficientul de determinare):** măsoară cât de bine sunt explicate variațiile țintei. Valoarea TEST de **0,9986** este foarte ridicată. R² nu este procent de predicții corecte și nu trebuie prezentat ca „acuratețe de 99,86%”.
- **Q05 coverage:** proporția observațiilor pentru care ținta este mai mare sau egală cu limita inferioară Q05. Pe TEST, aceasta este **97,06%**.
- **Interval coverage:** proporția observațiilor pentru care ținta se află între Q05 și Q95. Pe TEST, aceasta este **92,20%**, față de nivelul nominal de 90%.
- **Interval width:** lățimea medie Q95 − Q05. Pe TEST este de aproximativ **39,27 A**.
- **Quantile crossing:** frecvența cu care estimările brute ale cuantilelor nu sunt în ordinea Q05 ≤ Q50 ≤ Q95 și trebuie reparate prin sortare. Pe TEST, crossing-ul apare în aproximativ **26,45%** dintre observații.

Pentru o problemă de regresie, MAE și RMSE sunt indicatori direcți ai erorii predicției centrale. R² este un indicator complementar. Coverage și lățimea intervalului descriu comportamentul estimărilor de incertitudine.

## 6. Analiza generalizării și a overfittingului

Diferențele dintre MAE-ul de antrenare și cel de evaluare sunt:

| Comparație | Diferență MAE | Creștere față de Train |
|---|---:|---:|
| Validation − Train | 0,9189 A | 22,90% |
| Test − Train | 0,8179 A | 20,38% |

MAE crește de la 4,0127 A pe Train la 4,8306 A pe Test. Această degradare este vizibilă, dar performanța pe Validation și Test este apropiată, ceea ce este un semn favorabil de generalizare temporală.

Nu se poate concluziona doar din aceste valori că modelul nu are deloc overfitting. Totuși, diferența dintre Train și seturile nevăzute nu sugerează o prăbușire a performanței în afara setului de antrenare.

## 7. Backtesting temporal

Pipeline-ul include un backtest cu patru fold-uri de tip rolling-origin. În fiecare fold, antrenarea folosește date anterioare perioadei evaluate.

| Fold | Număr train | Număr test | Perioadă evaluată (UTC) | MAE (A) | Frecvență supraestimare Q05 calibrat |
|---|---:|---:|---|---:|---:|
| 0 | 42.075 | 13.077 | 2024-01-02 12:00 – 2024-07-02 02:00 | 5,1753 | 1,81% |
| 1 | 52.596 | 13.077 | 2024-07-03 03:00 – 2024-12-31 17:00 | 4,7671 | 3,77% |
| 2 | 63.114 | 13.077 | 2025-01-01 18:00 – 2025-07-02 08:00 | 4,6691 | 2,61% |
| 3 | 73.632 | 13.077 | 2025-07-03 09:00 – 2025-12-31 23:00 | 4,5837 | 3,75% |

Rezultatele backtest-ului arată o performanță relativ stabilă pe perioade succesive. Frecvența de supraestimare a limitei inferioare Q05 calibrate este între aproximativ 1,81% și 3,77% în aceste fold-uri.

## 8. Intervalele de încredere bootstrap pentru setul TEST

Raportul de antrenare include intervale bootstrap pentru anumite subseturi. Acestea oferă o estimare a variabilității metricilor pe blocuri temporale.

| Subset | N | MAE (A) | Interval bootstrap MAE (A) | Frecvență supraestimare Q05 |
|---|---:|---:|---:|---:|
| Toate observațiile TEST | 26.226 | 4,8306 | 4,6666 – 5,0392 | 2,94% |
| Zi | 13.178 | 6,5327 | 6,2904 – 6,7826 | 3,22% |
| Noapte | 13.048 | 3,1115 | 2,9818 – 3,2678 | 2,67% |
| Regim moderat: T ≥ 28 °C și vânt < 2 m/s | 250 | 9,2013 | 7,4536 – 11,4179 | 9,20% |
| Regim sever: T ≥ 30 °C și vânt < 1 m/s | 35 |  —  | Nu a putut fi estimat robust | — |

Subsetul sever are doar 35 de observații, insuficiente pentru un interval bootstrap robust în raportul curent. Performanța în acest regim trebuie interpretată cu prudență.

## 9. Exemplu de predicție

Un exemplu extras din setul TEST:

**Intrări:**

| Variabilă | Valoare |
|---|---:|
| Timestamp UTC | 2025-05-17 10:00 |
| Segment | SEG_1 |
| Temperatura aerului | 12,3 °C |
| Viteza vântului | 4,35 m/s |
| Radiație solară | 231 W/m² |

**Rezultate:**

| Rezultat | Ampacitate |
|---|---:|
| Țintă LineRate (simulată) | 1.361,57 A |
| Q05 | 1.353,49 A |
| Q50 – predicția centrală | 1.359,93 A |
| Q95 | 1.390,18 A |
| Eroare absolută Q50 | 1,64 A |

În acest exemplu, ținta LineRate se află în intervalul Q05–Q95.

## 10. Limitări și riscuri de interpretare

1. **Etichete simulate, nu măsurători reale.** Modelul a fost antrenat și testat față de rezultate generate de LineRate/CIGRE TB 601. Performanța raportată demonstrează aproximarea modelului ML față de acel motor fizic, nu validarea pe date operaționale reale.
2. **Incertitudinea intervalelor este limitată.** Q05/Q50/Q95 descriu în principal eroarea de aproximare a modelului ML față de funcția fizică deterministă. Nu includ integral:
   - eroarea prognozei meteorologice;
   - incertitudinea parametrilor conductorului;
   - variația produsă de săgeată, îmbătrânire sau starea reală a conductorului;
   - eroarea de formă a modelului fizic;
   - diferențele dintre simulare și măsurătorile dintr-o rețea reală.

   În exploatare, incertitudinea totală poate fi mai mare decât intervalele raportate aici.
3. **Quantile crossing.** Înainte de reparare, modelele cuantile independente produc crossing în 26,45% din observațiile TEST. Sortarea asigură ordinea finală, dar poate schimba rolul valorilor brute, inclusiv valoarea mediană. Acest comportament merită urmărit și îmbunătățit.
4. **Performanță mai slabă în anumite regimuri.** MAE bootstrap pentru subsetul de zi este 6,53 A, față de 3,11 A noaptea. În regimul moderat definit în raport, MAE este 9,20 A. Aceste diferențe indică necesitatea unei evaluări atente pe condiții dificile.
5. **Eșantion redus pentru regimul sever.** Doar 35 de observații TEST satisfac simultan T ≥ 30 °C și vânt < 1 m/s. Nu există suficient suport pentru concluzii ferme despre performanța în acest regim.
6. **Extrapolarea în afara domeniului de antrenare.** Modelele de tip arbore pot extrapola slab în afara intervalelor observate la antrenare. Pipeline-ul verifică intervalele caracteristicilor și poate emite avertismente; predicțiile în afara domeniului trebuie tratate cu precauție.
7. **Nu este încă validat pentru decizii operaționale.** Înainte de utilizarea într-o rețea reală sunt necesare comparații cu măsurători reale, validare inginerească, analiză de siguranță și proceduri de fallback.

## 11. Utilizarea modelului

Modelul salvat este disponibil în bundle-ul `models/ampacity_model.joblib`.

Funcția de inferență din proiect este:

```python
predictions = mm.predict_ampacity(
    weather,
    bundle=bundle
)
```

Datele de intrare trebuie să conțină coloanele meteorologice și identificatorul segmentului cerute de funcția `validate_weather(...)`. Timestamp-urile trebuie să poată fi interpretate, iar segmentul trebuie să existe în configurația bundle-ului.

Rezultatul conține, printre altele:

- `q05_ampacity_a`
- `q50_ampacity_a`
- `q95_ampacity_a`
- `static_rating_a`
- `uncertainty_ratio`
- `data_quality`

Rândurile invalide sau cu segmente necunoscute nu sunt completate prin presupuneri: predicțiile lor sunt NaN și `data_quality` indică `invalid_input`.

## 12. Metadate și reproductibilitate

Metadatele bundle-ului final:

| Câmp | Valoare |
|---|---|
| Data creării (UTC) | 2026-11-07 18:57:15 |
| Versiune scikit-learn | 1.6.1 |
| Versiune NumPy | 2.4.6 |
| Versiune pandas | 2.2.3 |
| Număr observații de antrenare | 52.596 |
| Sursa etichetelor | `LineRate.Cigre601 (CIGRE TB 601), SIMULATED conductor` |
| Hash dataset | `3cbfbc514ed2176508387180d1488d25f14fa6ce4f2ce01bf117bd477b31ac6f` |

Hash-ul identifică setul de date asociat antrenării și poate fi folosit pentru a verifica dacă datele utilizate ulterior sunt aceleași.

---

## Concluzie

Modelul GridPilot AI obține pe setul temporal de TEST un MAE de **4,83 A**, un RMSE de **7,27 A** și un R² de **0,9986**. Intervalul Q05–Q95 conține **92,20%** dintre țintele simulate, iar limita inferioară Q05 este sub sau egală cu ținta în **97,06%** dintre observații.

Diferența dintre MAE-ul de antrenare și cel de test este de aproximativ **0,82 A**, iar rezultatele de validare și test sunt apropiate. Aceste rezultate susțin faptul că modelul aproximează bine motorul fizic folosit pentru generarea etichetelor, pe perioade temporale nevăzute în antrenare.

**Concluzia trebuie formulată în limitele datelor disponibile:** modelul este validat experimental față de etichete fizice simulate LineRate/CIGRE TB 601, nu față de măsurători reale dintr-o rețea electrică. Validarea pe date reale rămâne un pas necesar înaintea oricărei utilizări operaționale.
