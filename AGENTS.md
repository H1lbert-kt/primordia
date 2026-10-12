# Primordia

Simulação 2D de ecossistema em que criaturas com redes neurais evoluem por
seleção natural. **Não existe função de fitness**: quem sobrevive e acumula
energia se reproduz. Predadores, bandos, fuga e emboscada devem *emergir*,
nunca ser programados.

Objetivo do repositório: ser um projeto de GitHub impressionante e honesto,
com demo visual, experimentos reproduzíveis e resultados documentados.

## Idioma
- Converse com o usuário em **português**.
- Código, identificadores, comentários, docstrings, mensagens de commit e
  README principal em **inglês**.

## Estado atual do repositório (verificado em 2026-10-12)
- **Etapa 18 ("defaults que sobrevivem", emergência de balanceamento)
  concluída, commit pendente de `/cp`:** plano em
  `.opencode/plans/etapa-18-balanco-defaults.md` (decisões do usuário:
  casa sai do default / próxima feature = etapa 19 daemon+SQLite).
  **Problema medido:** a Config default da CLI morria — seeds 42/7/13 ×
  5k terminavam com **3/86/70** vivos (fome 524, 27 nascimentos na 42,
  `n_house`→383): `house_energy_cost=20` **nunca foi gate-ada** (a 17
  calibrou só variações do `proto_culture.toml`). Isolamento seed42/5k:
  sem construção 327 ✓ / só fogo 195 / fogo+casa 3; documental
  `default+house=12` também colapsa (final 11, 425 casas).
  **Mudanças (só Config, zero tick):** `house_energy_cost` 20→**1e6**
  (proibitivo documentado; experimentos override — a 17 já fazia);
  `fire_energy_cost` 10→**5** (sweep pré-registrado {5,10,20} × seeds
  42/7 × 5k, regra = maior pop final média com cook>0: **408/145/60**;
  fogo barato vence até o mundo sem construção — cook devolve mais do
  que o build drena; **limite do bracket: 5 é o fim barato**, se
  rebaixar de novo refazer o sweep). **Gate (candidato × 3 seeds ×
  10k): PASS duro** — final **1163/1170/821**, min alive 101/110/137,
  births 2489–3054, cooked 45k–94k, casas 0 (esperado). Candidato ==
  default pós-edit (bit-idêntico, sem re-run). **Testes: 206 passed** —
  fix: `make_config` do `tests/test_civilization.py` fixa
  `house_energy_cost=20` (testes constroem casa com energia 60);
  `test_build_fire_requires_full_payment` agora resolve o caso "curto
  de energia" relativamente a `fire_energy_cost` (era pin 9.9 fixo).
  **Bench** (2 amostras, load 5–7): **277–579 / 175–265 / 113–148** a
  500/2000/5000, `alive_final` 109/662/1112 — piso 30 @N=2000 com
  5–8× (era 92–136 na 17 sob load 7; tick intocado). Docs:
  `experiments/civilization/README.md` ganhou **Postscript (stage 18)**
  com todos os números; `run.py` do experimento ganhou nota de
  rodapé (pré-registrado intacto); README principal bench 18 +
  reprodutibilidade + roadmap 1–18. **Próximo passo do usuário:
  `/cp`**, depois etapa 19 (daemon + SQLite).
- **Etapa 17 ("civilização": madeira, fogo, casas) concluída, commit
  pendente de `/cp`:** plano em `.opencode/plans/etapa-17-civilizacao.md`
  (decisões do usuário: 2 gates separados `build_fire`/`build_house`,
  +1 input `smell wood`, piloto + gate 3 seeds × 10k avisando;
  **`house_energy_cost=12`** para o gate; publicar **controle `off` +
  tratamento `h12` lado a lado**). **Núcleo:** novo recurso `wood (N,)`
  com `inventory (N,)` (pack size = `max_inventory=4`), campo
  `wood_field (H,W)` (value-noise patches + crescimento, RNG pós-spawn
  desloca trajetórias vs etapa 16), estruturas com free-list
  (`struct_fire/struct_house` no grid, `struct_free_count`,
  `max_structures=1024`); sensores: `smell wood` em `3·n_rays+13`
  (soma/picos/antenas da janela de cheiro, self-excluded);
  **`N_OUTPUTS` 4→8** (gather, build_fire, build_house; scaled-sigmoid);
  **layout 35 in / 8 out / 9 hidden / 485 brain_params / 490 genoma**;
  fases 15→**21** (`grow_wood`, `decay_structures`, `gather`, `build`,
  `rebuild_struct_fields` + `eat`/`bite`/`deaths` estendidos).
  **`FORMAT_VERSION` 8→9** (`inventory, wood, wood_field, struct_*,
  struct_free_count, wood_gathered, energy_cooked`; v1–v8 → ValueError
  "format"). **Config (defaults):** `wood_capacity=2.0,
  wood_growth_rate=0.01, initial_wood=0.0, wood_patch_amplitude=1.0,
  wood_patch_scale=60.0, gather_rate=0.5, max_inventory=4.0,
  fire_wood_cost=3.0, fire_energy_cost=10.0, house_wood_cost=4.0,
  house_energy_cost=20.0, fire_initial_fuel=3.0, fire_fuel_decay=0.002,
  fire_radius=6.0, house_radius=5.0, cook_bonus=1.5,
  cover_reduction=0.5, max_structures=1024` + **validação cruzada**
  `fire/house wood cost <= max_inventory` (plano trazia 6.0 > 4.0 =
  casa inafordável por construção; corrigido p/ 4.0 com guard).
  Semânticas: pagamento de build all-or-nothing; fogo construído cozinha
  no mesmo tick; cook nunca cria energia (extra drenado 1:1 do fuel);
  cover = escudo `land = take·(1−cover_reduction·cover)` com orçamento
  gasto inteiro; refuel `min(inventory, fire_initial_fuel−fuel)` sem
  custo de energia; uma estrutura por célula (fogo primeiro);
  `struct_fire` = menor slot; casas nunca decaem; mundos em sequência.
  **Stats:** `n_fire`, `n_house`, `wood_total`, `wood_gathered`,
  `energy_cooked` (+ fix do sigmoid duplo que somava 2×). **Render:**
  overlay de madeira, sprite de casa, painel 8 saídas `W2|b2 (8x10)`,
  linha `smell wood` + linhas `wood/fire/house`, HUD overview.
  **Testes: 206 passed** — `tests/test_civilization.py` (22: layout,
  validação, grow/gather/build/refuel/burnout, heat/cover, precedência de
  slot, cook ×3, cover bite, smell wood, stats, determinismo sequencial,
  save/load estruturas), `test_io` v8, pins atualizados (35/485/490/8,
  21 fases, 8-tupla biology, conservação com `wood_growth_rate=0.0`,
  painel `test_panel_shows_all_gates_and_w2_layout`).
  **Bench** (load 6.9–7.4): **127–134 / 92–136 / 42–124** a
  500/2000/5000 — piso 30 @N=2000 ✓; fases novas ≈138 µs ≈ **1.2%**
  (grow_wood 11 / decay_structures 4 / gather 33 / build 44 /
  rebuild_struct_fields 47). Screenshots `/tmp/opencode/
  stage17_overview.png` e `stage17_selected.png`.
  **Experimento `experiments/civilization/`** (proto_culture.toml,
  seeds 42/7/13, pre-registrado no docstring do `run.py`):
  **piloto** 2 seeds × 5k × {off, h20, h12, h8}: final 66/2/24/14
  (off seed42 recupera 74→110); H1 parcial **não-monótona** (h8 = spam
  591 casas, pior que h12); H2 ✓ (fire/cook ativos em todas).
  **gate** 3 seeds × 10k: **controle `off` (house=1e6) HARD GATE PASS**
  (434/110/243, frac 0.083, 804 births) vs **`h12` HARD GATE FAIL**
  (seed13 **extinta @8317**; 31/40/0; 169 births; frac 0.129) — casas
  derrubam a pop ~10×; `n_house` **estagna após ~5k** (501→501) =
  seleção fecha o spam, tarde demais; medido ✓ em h12 (fires 3/3,
  houses 3/3, cook/gather). Controle rodo **pós-h12** (divulgado no
  README). `experiments/civilization/README.md` com números, figs
  pilot/gate (economy/building/wood) e 5 limitações. README principal:
  490/35/206 testes/bench 17/roadmap 1–17 + `civilization` na lista.
  GIF demo não regenerado (não pedido). **Próximo passo do usuário:
  `/cp`**, depois daemon/território/Oja.
- **Etapa 16 ("doação") concluída, commit pendente de `/cp`:** plano em
  `.opencode/plans/etapa-16-doacao.md` (decisões do usuário: receptor =
  **vizinho mais próximo** em `donate_range` (recepção automática, sem
  gate do receptor), `hidden_size` **10→9** p/ caber no orçamento,
  experimento **só `experiments/proto_culture/`**, toml
  `signal_emit_cost` 0.5→**0.05**). **Núcleo:** `Config.donate_rate=0.0`
  (**default off** — sigmoid nunca fecha, preço default taxaria todos;
  validações `>=0`) + `Config.donate_range=3.0`; `N_OUTPUTS` 4→**5**
  (gate `[4]`, scaled-sigmoid em `apply_actions`); **layout 34 in / 5
  out / 9 hidden / 446 brain_params / 451 genoma**; fases 15→**16**
  (nova `donate` após `emit_signal`, antes de `eat`; docstring do tick
  renumerada). Kernel `_donate` serial sem RNG (espelha o mate search:
  células CSR no alcance, d² toroidal exata, self excluído, empate →
  menor slot), `t=min(donate_rate·gate, energy[i])`, conservação exata
  (teste approx float32), retorno `(total, align_sum, pairs)` com
  alinhamento = dot dos vetores unitários de hue. **`FORMAT_VERSION`
  7→8** (`energy_donated` float64 salvo; v1–v7 → ValueError; docstring
  stale "v6" corrigida). **Scratch (não salvos):** `donate_to (N,)
  int32` (mapa doador→receptor p/ render, fill(-1) a cada tick,
  custo 0.75 µs no caminho default) + `donate_align_sum/donate_pairs`.
  **Stats:** `donated` (delta), `donate_mean` (gate médio vivos),
  `donate_align` (mean dos pares, 0 sem pares). **Render:** linhas
  cyan `flash_donation` donor→receptor via `EventTracker.transfers`
  (idade por doador, fade `flash_duration`, **skip de wrap do torus**
  — artefato pego no screenshot), painel ganha linha `don` +
  `W2|b2 (5x10)`; `plot.py` intocado. **Testes: 183 passed** (18 novos:
  `tests/test_donation.py` 12 — conservação, no-op, nearest/empate,
  clamp, faixa, fórmulas 5/446/451, determinismo, gift map, continuation
  v8 bit-a-bit; v7→ValueError em `test_io`; colunas stats; painel
  `don`; transfers lifecycle + wrap + alpha em `test_dynamics`); pins
  494/499→446/451, 15→16 fases. **Bench** (load 1.1–4.6, amostras
  oscilam 2–4× — 2ª amostra N=5000 > N=2000 = ruído): **965–1374 /
  349–577 / 139–352** a 500/2000/5000; fase `donate` medida isolada:
  0.75 µs (rate=0) / ~24 µs (ativa ≈ reproduce). **Experimento
  `experiments/proto_culture/`** (off/cheap 0.02/dear 0.10 sobre toml
  0.05, seeds 42/7/13, piloto 800 + gate 10k, caches em `data/`):
  **HARD GATE PASS** (9/9 vivos, min 121–137, pops finais 845/827/887
  médias); **H1 refutada em cheap** (gate 0.708→0.737 *subiu*) e fraca
  em dear (0.706→0.666, −6%); **H2 confirmada**: align 0.511/0.540 vs
  chance `(1−hue_spread)²` 0.108/0.208 = **4.7×/2.6×** (gifts p/ parentes;
  figura suavizada 201-tick); context: pop/energia ~igual controle,
  predation 994/650/971 e barrier 0.092/0.032/0.071 divergem mas n=3 =
  noise (documentado). README do experimento com 5 limitações; README
  principal: 451/34/183/bench 16/roadmap 1–16 + `proto_culture` na
  lista de reprodução. GIF demo não regenerado (default = doação off,
  visual idêntico à 15). **Próximo passo do usuário: `/cp`**, depois
  civilização/daemon/território/Oja.
- **Etapa 15 ("tag/cor") concluída, commitada em `origin/master`
  (`f2f4554`):** plano em
  `.opencode/plans/etapa-15-tag-cor.md` (decisões fechadas pelo usuário:
  5º traço `tag_hue ∈ [0,1)`, **sem campo no chão**, **1 input** de
  similaridade lendo vizinhos diretos na janela de cheiro (self-excluded),
  sprite **mantém matiz=dieta** + **anel fino colorido** sob o sprite +
  swatch no painel, `FORMAT_VERSION` 6→7 **confirmado**, experimento
  3 seeds × 10k **confirmado**). **Núcleo:** spawn
  `genome[traits+4]=rng.uniform(0,1)`; mutação **circular**
  `h' = h + N(0,trait_std)` com wrap `h − floor(h)`; crossover uniforme
  (filho herda cor de 1 pai → parente reconhecível); `rebuild_counts`
  pré-computa `hue_re/hue_im (N,)` (np.cos/sin vetorial) e acumula
  scratch `cell_tag_re/im (H,W)` (**não salvos**) no `_rebuild_grid`.
  **Sensor:** `(Σre·own_re + Σim·own_im − 1)·inv_smell ∈ [−1,1]`
  (sozinho=0, multidão idêntica≈1), índice `3·n_rays+12 = 33`.
  **Layout: 34 in / 4 out / 494 brain_params / 499 genoma** (limite 500);
  `N_BODY_TRAITS` 4→5; `sensor_input_dim = n_rays*3+13`.
  **`FORMAT_VERSION` 6→7** (v1–v6 → ValueError; layout do genoma mudou).
  **Stats novos:** `hue_mean` (circular), `hue_spread` (1−R; 0=cor única,
  1=roda uniforme), `tag_sim_mean`, `win_neigh` (= nº de vizinhos na
  janela = leitura crowd do sensor × cells², reaproveita `sensor_buf`,
  custo ≈0); `plot.py` intocado. **Render:** `hue_lut()`, anel de 8
  offsets sob o sprite (pass antes do humanoide, `_scatter_groups`
  agrupa por cor), painel `tag 0.42` + swatch 12 px na linha vision/diet
  (gaps de seção trimados p/ caber 720 px — `test_panel_fits_window`
  ajustado). **Testes: 165 passed** — `tests/test_tag.py` novo (10:
  spawn/mutação circular+wrap, similaridade exata, layout/input_groups,
  stats, save v7, determinismo); pins 34/494/499 em `test_brain`/
  `test_signals`; `test_io` + `test_v5`/`test_v6_save_is_rejected`;
  fix `test_determinism_with_predation` (layout reshuffled o stream do
  Numba → vieses de gate `genome[a,b2+2]=20`, `genome[v,b2+2]=−20`,
  `b2 = hid*in + hid + out*hid`). **Bench pós-mudança** (2 amostras,
  load 4.3–4.9): **922–1250 / 464–490 / 258–287** a 500/2000/5000 vs
  baseline 14 (794–1106/465–491/166–303, sessão anterior) → sem
  regressão, piso 30 @N=2000 com 15×. **Validação sensor** (mundo
  tick-3000): série direta == `sensor_buf` (maxdiff **0.00e+00**),
  n̄ vizinhos 0.79 vs 0.06 aleatório. Caches npz de experimentos
  antigos ficam staleness pelas colunas novas (re-run automático, não
  bloqueia). Screenshots: `/tmp/opencode/stage15_overview.png` e
  `stage15_selected.png` (anel + `tag` no painel + `W1 (10x35)`).
- **Experimento `experiments/tags/` (etapa 15)** — default Config,
  seeds 42/7/13 × 10k, ~214–253 ticks/s. **HARD GATE PASS** (3/3 vivas;
  min alive 55/54/141, final 270/469/939, média 178/189/363). **H1
  deriva neutra ✓:** `hue_spread` final **0.175/0.773/0.796**
  (semente-dependente, não monotó; banda abre após gargalo ~6500) e
  `hue_mean` circular circula/abota (wrap visível). **H2 estrutura ✓:**
  `tag_sim_mean` médio 0.046/0.054/0.042, final **0.150/0.132/0.171**
  vs nulo_B (posições+cores aleatórias) final ≈0.0005–0.005 e nulo_A
  (cores aleatórias dado o espaçamento) ≈0.009–0.10. Decomposição:
  **spacing factor** final **22.4/17.5/9.4** (inicia ≈1.0 ✓; média
  pós-1k = 17.9/23.1/10.1; pico ~140 no gargalo) e **align ratio**
  (observado/nulo_A) final **1.5/12.5/18.8**. **H3 não afirmada**
  (estrutura ≠ uso do sensor; ablação exigiria layout diferente —
  futuro). Figuras `drift/similarity/pop.png`; README com 5
  limitações (ratio ruidoso cedo p/ R²≈0; seed42 diluído; nulo_A
  aprox.; pilot informou H2 — divulgado). `pop.png` saudável (energia
  30→65–90, sem colapso). README principal: genoma **499** / **34
  inputs** / **165 testes** / bench 15 / roadmap 1–15 / `tags` na lista
  de reprodução. **Próximo passo do usuário: `/cp`**, depois planejar
  a etapa 16 (doação).
- **Etapa 14 ("feromônio/difusão") concluída, commitada em
  `origin/master` (`3552e4e`):** Jacobi 5 pontos sobre o campo `signal` com wrap toroidal
  (`new = old + rate·(N+S+E+W−4·old)`), conserva massa por construção e é
  estável só p/ `rate ≤ 0.25` (validado em `__post_init__`). Novo
  `Config.signal_diffuse = 0.1` **ligado por default** (decisão do usuário:
  física livre, ao contrário do emit cost) e 0 = no-op exato (controle do
  experimento). Kernel escreve no novo scratch `World.signal_buf (H,W)
  float32` e a fase copia de volta (`np.copyto` — cópia, não swap, para
  referências de `signal` ficarem válidas). Fase nova `diffuse_signal` logo
  **após `decay_signal`** (fases 14→15; docstring do tick renumerada):
  decay+difusão evoluem o campo junta, as fontes (emit) ficam depois —
  contrato "emit só visível no tick seguinte" intocado. **`FORMAT_VERSION`
  fica 6** (sem array novo no save; precedente = campos só-Config da 9.5;
  caveat: continuar save da 13 aplica o default 0.1 — muda a física no
  meio da run, documentado). Layout **inalterado 33 in / 4 out / 484 / 488**
  (difusão pura, sem sensores novos). **Stats novo:** `signal_cells` =
  nº de células > `0.01·signal_capacity` (área da trilha; `_SIGNAL_EPS`
  constante de medida em `stats.py`, fora do tick; `plot.py` intocado).
  Docstring de `world.py` corrigida (`signal`/`signal_buf` ausentes e
  `actions (N,3)` defasado desde a 13 → agora (N,4)). **Testes: 153
  passed** — `tests/test_diffusion.py` novo (conservação de massa, spread
  com wrap, estabilidade 100 passos a 0.25, no-op em 0, ordem de fases,
  validação, determinismo e continuação save/load bit-a-bit com difusão
  forte a 0.25).
- **Experimento `experiments/trails/` (etapa 14)** — duas fases sobre
  `proto_culture.toml`, seeds 42/7/13. **Fase price** (6k,
  `signal_diffuse=0` p/ isolar a economia): custo {0, 0.01, 0.05, 0.15} →
  pop final 219/212/209/99 com **3/3 sobreviventes em todos**, `emit_mean`
  0.717→0.579 e emitido −77% (seleção age no gate). Regra estrita do plano
  escolhia **0.15** (maior custo sobrevivente), mas terminou 6k a 45% do
  controle (min 71, descendente) com risco real de extinção a 10k —
  **decisão do usuário: gate a 0.05** (pop 209 ≈ controle 219, seleção já
  visível). **Fase gate** (10k, preço 0.05; nodiff 0 / mid 0.1 / fast 0.2):
  duro ✓ (todas 9 runs vivas, min 290 → **janela de preço da 13 fechada:
  0.05 é viável até 10k**); física ✓ `signal_cells` final 7740 → 15439 →
  16529 (~2×; a meio do run ~3.4×; nodiff nunca passa 11.7k células = 29%
  do grid, mid/fast atingem 34–37k = 85–94% do grid); comportamento
  **nulo como previsto a priori** (pops 974/859/900 e predação 469/417/485
  com bandas sobrepostas, `diet_max` misturado). `signal_mass` difere entre
  variantes (4.4k–5.6k) pela *fonte* (população × gate), não por criação
  de massa — a conservação é teste unitário. **Bench:** fase
  `diffuse_signal` = **72 µs ≈ 3.2%** do tick (1523→2269 µs sum, profile);
  completo (3 amostras, load 3.4–5.6): **794–1106 / 465–491 / 166–303** a
  500/2000/5000 — piso 30 @N=2000 com 15× (faixa 13 era 683–1346/516–861/
  162–516 em load menor; diferença = ruído de carga, share da fase medido
  no profile). README principal: 153 testes (superado pela atualização
  da 15), tabela de bench, roadmap 1–14 + nota de reprodutibilidade da 14.
- **Etapa 13 ("canal de sinais") concluída, commitada em
  `origin/master` (`e0fb14b`):** canal comum escalar **sem recompensa** (princípio 1 intocado): campo
  `signal (H,W) float32` com decaimento geométrico **local** (sem difusão —
  essa é a etapa 14), gate de emissão = 4ª saída do cérebro (`N_OUTPUTS`
  3→4, mesmo scaled-sigmoid do eat gate), depósito `emit_rate·gate` na
  própria célula e custo `signal_emit_cost·gate` drenado da energia;
  sinal emitido só visível no tick seguinte (perceive roda antes de
  emit_signal — contrato documentado em `step.py`). **3 sensores novos**
  (`n_rays*3+9…+11`): cheiro do campo na janela de cheiro + antenas
  esq/dir por produto cruzado (sem self-exclusion — auto-monitoramento
  permitido). Layout: **33 in / 4 out / 484 brain_params / 488 genoma**;
  fases 12→14 (`decay_signal` após `decay_meat`, `emit_signal` após
  `apply_actions`). **`FORMAT_VERSION` 5→6** (`signal` + `signal_emitted`;
  v1–v5 → ValueError). Novos `Config`: `signal_decay=0.05`,
  `emit_rate=1.0`, `signal_emit_cost=0.0`, `signal_capacity=1.0`.
  **`signal_emit_cost` default = 0 é desvio deliberado do plano (0.5):**
  o gate sigmoid nunca fecha, então um custo default >0 taxaria toda
  criatura ~cost/2 por tick desde o nascimento e quebraria conservação e
  balanço; o preço é tratamento de experimento (em `proto_culture.toml`
  = 0.5). **Stats novos:** `signal_emitted` (delta), `signal_mass`
  (soma do campo), `emit_mean` (gate médio sobre vivos). **Render:**
  overlay violeta `signal_color` por último no blend, HUD `sig N`,
  painel com linha `emit` (pitch ajustado p/ caber em 720 px). **Experimento
  `experiments/signals/`** (proto 0.5 / cheap 0.001 / pricey 2.0 sobre
  `proto_culture.toml`, 3 seeds, screening 6k + gate 10k): hipótese
  a priori = preço é contornado pelo fechamento do gate sem mudar pop
  — **metade refutada**. `emit_mean` cai monótono com o preço (0.70 →
  0.33 → 0.00), seleção age no gate em poucos k ticks, **mas** o imposto
  bate desde o nascimento (gate ~0.7 × 0.5 = 0.35 e/tick ≈ 4× metabolismo)
  e a pop morre antes: gate 10k = proto 33 vivos (seed 13 extinta),
  cheap 728 (spam 2M unidades, predação 788, diet_max 0.52 — só a
  variante barata tem predação viva), **pricey extinta 3/3**. Gate duro
  ✗ p/ proto e pricey, ✓ p/ cheap; medido (emitted>0 em ≥1/3) ✓ mas
  irrelevante (mundo pagando estava morrendo). Janela de preço viável
  não caçada (só bracket). `proto_culture.toml` ganhou
  `signal_emit_cost=0.5`; **defaults de `Config` não mudaram**.
  **Suíte: 143 passed** (`pytest -q`, 151 s). Bench (load 0.8–3.8, duas
  amostras): **1346/861/516** e **683/516/162** a 500/2000/5000 — piso 30
  @N=2000 com 17–29×, sem regressão vs 12.5 (1133/788/310; a diferença é
  ruído de carga, o tick novo custa ~2 leituras por célula no smell). `plot.py` ganha
  fig `signal.png` com guard por presença de `signal_emitted`.
  Seguiu a etapa 14 (difusão, plano `.opencode/plans/etapa-14-feromonio.md`);
  o sweep de preço virou a fase "price" de `experiments/trails/` (0.05
  viável a 10k), como o "calibrar antes da 14+" previa.
- **Etapa 12.5 ("ratchet de predação") concluída, commitada em
  `origin/master` (`9646ab5`):** ratchet **quebra com Config só** (nada de
  tick novo além de contadores de diagnóstico). `Config.bite_rate`/`contact_range`/`bite_efficiency`
  são as alavancas; experimento em `experiments/predator_survival/`:
  screening 8 variantes × 3 seeds (42/7/13) × 6k + gate `--phase gate`
  (base/reach/bigbite+ × 10k). Resultado: `reach` (bite_rate=12,
  contact_range=2.5) = 1179 mortes por predação/run vs 43 do base a 10k;
  `diet_max` final médio 0.394 (max 0.495 seed 42), `diet_p50` 0.207 —
  mas **`bite_pred = 0` em todas as runs**: nenhum atacante com
  diet>0.66 completa mordida (diluição por crossover, previsão a priori
  confirmada). Gate estrito (diet>0.5 em 3/3) ✗; predação ≥100/run ✓.
  **Novas colunas de stats** (`primordia/stats.py`): `diet_p50/p90/p99`
  (percentis amostrados com genome_dist, forward-fill) + `bite_herb/omni/pred`
  (deltas por tick de `World.bite_by_diet`, buckets por dieta do atacante
  <0.33/0.33–0.66/>0.66). `World.bite_by_diet` (3,) int64 — diagnóstico,
  **não salvo** (sem bump de FORMAT_VERSION). `_bite` em `step.py` ganhou
  3 params (`bite_by_diet`, `herb_max`, `pred_min`); bench **1133/788/310**
  a 500/2000/5000 (piso 30 @N=2000 com 26×). **Testes: 133 passed.**
  **TOML:** `load_config` aceita `.toml` (tomllib stdlib, Python ≥3.11) além
  de `.json`; mensagem "expected an object of Config fields"; teste novo em
  `tests/test_config.py` (12 passed). **`configs/proto_culture.toml`**:
  pin da variante top `reach` (bite_rate=12, contact_range=2.5) para as
  etapas 13+; defaults de `Config` **não mudaram** (reprodutibilidade).
  `run_variant` staleness inclui colunas ausentes (npz antigo → re-run).
  Próximo passo do usuário: `/cp`, depois planejar etapa 13 (canal de
  sinais) em `.opencode/plans/etapa-13-sinais.md`.
- **Etapa 12 ("ecologia") concluída (2026-10-10, commit `1c71de9`
  em `origin/master`):** plano em `.opencode/plans/etapa-12-ecologia.md` (fases A–F).
  **Sem código de tick novo** — só métricas fora do tick, para medir o que
  10+11 deveriam produzir. `primordia/ecology.py` (novo): `classify_diets`
  (herbívora diet<0.33 / onívora / predadora diet>0.66, **só para contar,
  nunca regra**), `genome_scales` (dedup de `_genome_scales` do stats),
  `cluster_living` (greedy-leader determinístico por slot ascendente,
  ≤ threshold inclusivo, empate → líder menor; **não** single-linkage).
  `Config.species_threshold` (default **10.0**, calibrado: histograma de
  distâncias pareadas a 3000 ticks é bimodal, mediana ~7.5 com cauda até
  ~370 → K estável em `[8,50]`, caindo 5–15; 1.0 daria K≈N). Colunas novas
  no `StatsRecorder` (gauges **todo tick**: `n_herbivore`/`n_omnivore`/
  `n_predator`; amostradas a cada `cluster_every=100` e forward-filled:
  `n_species`, `barrier`, `matings_valid`). `barrier` = fração de nascimentos
  desde a última amostra com **os dois pais ainda vivos** e clusters
  diferentes (viés documentado: genitor de vida curta some do denominador —
  lower bound); `matings_valid` é o denominador exposto. **`partner_id` no
  log de genealógia: `genealogy` (L,3)→(L,4)**, `io FORMAT_VERSION = 4→5`
  (v1–v4 → ValueError). **Decisões do usuário:** (1) `partner_id` no log,
  (2) HUD count-only (sem colorir sprites por cluster; sprites mantêm
  matiz=dieta; sem GIF novo), (3) experimento 10k × 3 seeds. HUD: linha
  `species K` embutida na linha `tick/alive` + `season S(t)` na linha
  `seed/light` (**sem linha nova — `test_panel_fits_window` estouraria
  720 px**); app cacheia `cluster_living` a cada **30 frames** (O(N·K)
  Python, ~4–6 ms @N≈900, K=1); painel selecionado mostra **`cluster #k`**
  no lugar do vestigial `species 0` (nunca mais 0; `species_id` continua no
  mundo/save). **Experimento `experiments/ecology/`** (default vs
  `no_predation` = `bite_efficiency=0`, 3 seeds, 10k, figuras
  clusters/barrier/diet_classes/cross_corr + README com números reais):
  pop final 1703/1330, `n_species` 1→~18 monotônica, **barrier média
  0.100 (default) vs 0.057 (sem recompensa de carne)**, barrier>0 em
  85%/70% dos ticks pós-meio, `genome_dist` final 5.59/4.22, predação
  43/13 mortes (1.49%/0.46% do total; fome domina ~2800/run), troféu
  trófico ainda plano (≥99.9% herbívoro, 0 predadores em 10k). Resultado
  honesto: **especiação-lite real e mensurável, não requer predação
  (clusters formam sem carne) mas a recompensa de carne ~duplica a
  barreira**; sem ciclos Lotka-Volterra (energy→alive só na borda de 400
  ticks, r≤0.09). **`plot.py` ganha guard** por presença de `n_species`
  (npz antigo → 4 figs, sem crash; novo → `ecology.png`); `plot.py`
  antigo continua funcionando. `experiments/lineage/run.py` desempacota o
  log como 4-tuple. `run_variant` aceita dict ou `Config` pronto. **Testes:
  125 passed** (`pytest -q`, 23 s; +`test_panel_shows_cluster_when_cached`).
  **Bench** (load <2, duas amostras): **1409–1696 / 766–792 / 459–463** a
  500/2000/5000 — piso 30 @N=2000 com ~25×, sem regressão vs etapa 11
  (1397/627/292); o tick não mudou, só bookkeeping fora dele. Próximo
  passo do usuário: `/cp`.
- **Etapa 11 ("biologia") concluída (2026-10-09):** plano em
  `.opencode/plans/etapa-11-biologia.md` (fases A–F);
  decisões do usuário: reprodução **100% sexual** (sem fallback assexuado;
  colapso ⇒ calibrar com o usuário), 4º smell = carniça → layout **30 in /
  443 brain_params / 447 genome**, carne = predação deposita
  `carrion_yield·(1−bite_efficiency)·take` + morte por idade deposita
  `carrion_yield·energia` (fome ~0), `max_age` 2000→**5000**,
  `senescence_rate` default **0.0005** (sweep Fase E: 0.002 extingue,
  0.001 colapsa p/ 73–126, 0.0005 estável 277–417). Novos Config:
  `mate_range=3.0`, `senescence_rate=0.0005`, `carrion_yield=0.3`,
  `meat_decay=0.02`. Wear `w=exp(senescence_rate·age)`: metabolismo ×w,
  velocidade/visão `/w`. Busca de par em células CSR (distância toroidal
  exata decide, determinístico; empate → slot menor); crossover uniforme
  por gene + mutação; `child_e = child_fraction·energy[i]`, cada pai paga
  metade. Nova fase `decay_meat` (12 fases totais). Cheiro carrion =
  `meat/capacity/9` por célula vizinha. **io `FORMAT_VERSION = 4`**
  (`_ARRAYS` += `meat`, `_FLOAT_SCALARS` += `meat_eaten`; v1-v3 →
  ValueError). **Render:** `meat_color=(140,40,36)` em `Settings`, blend
  `rgb = lerp(lerp(terreno, comida, a_f), carne, a_m) × light` com
  `a_m=clip(meat/cap,0,1)`, HUD `mean energy … meat N`
  (`/tmp/opencode/stage11_meat_zoom.png`). **GATE 3 seeds (42/7/123) ×
  10k:** duro ✓ — sem extinção (min 158–189), pop média 590–747 < 1900,
  nascimentos 4599–6163 > 0, fracE 0.029–0.035 < 0.5; medido — predação
  **110/10/612** (≥2/3 ✓), `meat_eaten` 537–22877 ✓, `genome_dist` final
  3.49–6.45, idade na morte distribuída 0–3000 (sem cliff em 5000 —
  wear mata de fome antes do `max_age`; `deaths_age = 0` documentado).
  **Bench:** **1397/627/292** a 500/2000/5000 — piso 30 @N=2000 com 21×.
  GIF `docs/demo.gif` regenerado (9000 ticks, 26 frames, **1.18 MB**).
  `STAGE8_BASE`/README: reruns em HEAD já são sexuais (nota
  "Reproducibility"). README: genoma **447** / **30 inputs** / **109
  testes** / seção "Sex and aging". **Testes: 109 passed.** Próximo passo
  do usuário: `/cp`.
- **Etapa 10 ("ambiente físico") concluída (2026-10-09, commit pendente
  de `/cp`):** plano em `.opencode/plans/etapa-10-ambiente.md` (fases A–F);
  decisões do usuário: luz vira input → layout **29 in / 433 brain_params /
  437 genome**, amplitude do terreno = fração de terra (`n < 1−amp` →
  0.0 exato; `amp = 0` = terreno off sem desenhar RNG). **Leis físicas**
  puras em `primordia/cycles.py`: `L(t) = 1 − day_amp·0.5·(1 −
  cos(2πt/day_period))` ∈ [1−amp, 1], `S(t) = 1 + season_amp·sin(2πt/
  season_period)`, `growth_multiplier = S·L` cru (o *rate* aplicado em
  `phase_grow_food` é `min(rate·mult, 1.0)` — clampar o produto matava o
  boom sazonal, bug pego por teste). Terreno: `_make_terrain()` value-noise
  compartilhado com os patches (RNG após o spawn; amplitude 0 não desenha),
  água `terrain == 0.0`: não cresce comida e move custa
  `move_cost·|v|²·water_move_cost`. `perceive` ganhou `light`: visão =
  `visão·L(t)` (à noite encolhe) + 8º input. **io `FORMAT_VERSION = 3`**
  (terrain salvo; v1/v2 → ValueError; luz/estações não são salvas — são
  funções do tick). **Render:** `make_terrain_lut` (água/shore/verde),
  blend por célula `lerp(terreno, comida, comida/cap) × L` num passe
  200×200 por frame, sprites × L, overlays em brilho cheio, HUD `light`
  (`/tmp/opencode/stage10_overview.png`). **Fase E:** luz 0.75 ×
  fertilidade ~0.43 corta a oferta ~60% → sweep `food_growth_rate
  {0.04…0.27}` × seeds 42/7 × 6000 → default **0.02→0.12** (pop 1283–1471
  @6k, na faixa da 9.5). **GATE 3 seeds (42/7/123) × 10k:** duro ✓ — sem
  extinção (min 110–161), pop média 1058–1436 < 0.95·cap,
  frac(E≥limiar) 0.01–0.11 < 0.5; medido — fome **26–37%** (faixa 25–40
  ✓), ocupação `mean(terrain|vivas) − mean(map)` **+0.17…+0.22** (nicho ✓),
  correlação alimento×S(t) **r 0.39–0.59** (janela = 1.25 ciclos, honesto),
  comida de dia ~5% acima da noite (0.058 vs 0.057 vel. média), predação
  **1926/2/1** (seed 42: 0 até tick 7k, estouro de onívoros no boom final
  com pop no cap; nenhum dieta>0.5 vivo nos snapshots), deslocamento vs
  baseline 9.5 (mesma seed, terreno/ciclos off): **+20/−4/+45%** (misturado).
  Honestidade: pop *final* = 2000 (pico do serrilhano) em todos os runs do
  gate, e fome na marca 10k (26–37%) acima da faixa 11–27% da 9.5. **Bench**
  (load 5.6 → 2.6): **846/201/236** e **1533/700/430** a 500/2000/5000 —
  piso 30 @N=2000 com 6.7–23× (pop do bench nova: alive_final
  223/597/1203). GIF `docs/demo.gif` regenerado (9000 ticks, 26 frames,
  **0.99 MB** — frames noturnos comprimem melhor; orçamento <4 MB).
  `STAGE8_BASE` += `terrain_amplitude/season_amp/day_amp: 0` (reruns
  mantêm o regime da 8; staleness do cache é por nº de ticks, não
  invalida). README: genoma **437** / **29 inputs** / **92 testes** /
  bench etapa 10 / seção "A world that pushes back". **Testes: 92
  passed.** Próximo passo do usuário: `/cp`.
- **Etapa 9.5 ("o mundo precisa doer") concluída (2026-10-09):** plano em
  `.opencode/plans/etapa-9.5-balanco-e-pressao-seletiva.md`; roadmap
  "mundo real" aprovado em `.opencode/plans/roadmap-mundo-real.md`
  (10 ambiente → 11 biologia → 12 ecologia → 13–16 = antiga 10–13).
  **Diagnóstico que motivou** (baseline `4724fd0`, seed 42): mortes
  3984 idade/104 fome/**0 predação**, pop pinada no cap, 99% acima do
  limiar (seleção≈0), rotação ~90 rad/200 ticks (pião), sprite ~2 px.
  **Mudanças:** (1) economia recalibrada por sweep (seeds 42/7×6000):
  `food_capacity` 100→**3.0**, `initial_food` 60→**2.0**,
  `food_growth_rate` 0.05→**0.02**, `eat_rate` 4→**1.0**,
  `metabolic_cost` 0.05→**0.08** (oferta growth·K·células ≈ demanda →
  fome regula pop); (2) física do giro: novo `Config.turn_cost` = energia
  por radiano girado (`turn_cost*|tanh|*max_turn` debitado em
  `apply_actions`, kernel ganhou `energy`+`turn_cost`), default **0.3**;
  `max_turn` 0.6→**0.3**; (3) visual: `initial_zoom` 0.85→**1.5**,
  `size_scale` 2.4→**3.0**, `max_radius_px` 6→**8** (humanoide ~4–5 px;
  screenshot `/tmp/opencode/stage95_overview.png` — humanoids visíveis e
  **anéis de depleção** emergindo no campo de comida). **GATE 10k ticks
  (3 seeds):** pop mean/min 1342–1734/185–295 (abaixo do cap ✓),
  fome **11–27%** das mortes (faixa alvo 25–40% atingida a 6k/seed-42;
  a 10k dilui vs morte por idade — honestidade: Growth 0.015 sobe fome
  p/ 26% mas derruba frac≈0 e min pop 185; default mantido g.02),
  frac(E≥limiar) **0.01–0.32** (<0.5 ✓), **predação >0 nas 3 seeds**
  (1/5/64–141), rotação **9.9–14.2** rad (6–9× menor que baseline),
  retidão ainda 0.000 (pião mais lento, não eliminado — limitação
  honesta, próxima alavanca = terreno/estações etapa 10).
  **Testes: 76 passed** (3 novos: `test_turn_costs_energy`,
  `test_turn_cost_zero_is_free`, `test_turn_cost_validation`; 6 ajustados
  por herdar defaults — contabilizar `turn_cost` nos testes de
  conservação, `field.std()` relativo à capacity, prefix CSR de
  `cell_slots`, economia pinada em testes de IO/stats). **Bench**
  (load 2–5): **989/870/321** a 500/2000/5000 — piso 30 @N=2000 com 29×
  (tick mudou: +1 multiply no apply_actions). `STAGE8_BASE` em
  `experiments/_common.py` agora pinza **toda** a economia da etapa 8
  (capacity/eat/growth/metab/move/max_turn/turn_cost=0) — caches
  intactos. `docs/demo.gif` regenerado (9000 ticks, 26 frames,
  **3.69 MB** <4 MB). `overrides.example.json` atualizado (inclui
  `turn_cost`). Síntese: **seleção existe agora** (fome + giro caro +
  predação viva), mas o mundo ainda é um torus plano — etapa 10
  (terreno/estações/luz) é o próximo degrau de realismo aprovado.
- **Etapa 9 ("mundo que ensina") Fases A–F5 concluídas:**
  planos em `.opencode/plans/etapa-9-mundo-que-ensina.md` (+ roadmap
  10–13 do usuário). Ordem aprovada F1→F5. **F1** painel guiado:
  `input_groups(cfg)` em `sensors.py` (layout de entradas documentado e
  testado, guia "click a creature...", `overrides.example.json`,
  `--select` por creature id). **F2** `food_field` (campo contínuo
  200×200, patches com `food_patch_amplitude/size`, crescimento por
  célula em `_grow_food`, save **v2** = `food_field` + `format_version`);
  `STAGE8_BASE = {"food_patch_amplitude": 0.0}` em `experiments/_common.py`
  aplicado nos 4 `run.py` (caches intactos). GATE 5000 ticks aceito pelo
  usuário: energia cicla 1650–3650, famine 71–95/run, pop 2000,
  food_std 18.6 (patch) vs 9.8 (uniforme), ray_std 0.15–0.19,
  turn_std 0.51–0.72, predation 0; limitação honesta: patches sozinhos
  não eliminam os círculos em 5000 ticks. **F3** render: sprites
  humanoides (`_humanoid_offsets`, face norte + α = heading+90, agrupados
  por (raio,bucket)), gamma 0.75 na LUT de comida, trilha e flashes de
  morte (`EventTracker`/`Flash`/`draw_trail`, overlay SRCALPHA antes de
  `draw_overlay`), sparklines + legenda **só no overview**, auto-follow
  ao selecionar, fix: `clear_font_cache()` após `pygame.quit()` (2ª
  chamada de `run_app` quebrava). **F4** Elman + sentidos ricos:
  `hidden_prev (N,hid) float32` persistente (save v2 + v1 → `ValueError`),
  `think` com `W_rec` em `b2+N_OUT` (`hid²`=100 params), `hidden_size`
  default **12→10** p/ caber no orçamento → **28 in / 423 brain_params /
  427 genome_size** (meta 100–500); sensores `3*n_rays+7` (soma+**pico**
  por raio + antenas esq/dir por produto cruzado, `input_groups` com
  índice ndarray p/ "sum|peak"); painel: `W2|b2 (3x11)` +
  `W_rec (10x10)` lado a lado, W1 cell 10 (29 colunas). **Suíte 73
  passed** (`pytest -q`, 197 s). Evidências visuais: screenshots SDL
  dummy `/tmp/opencode/f4_selected.png` (painel completo com W_rec) e
  `f4_overview.png` (sparklines+legenda). **F5 concluída:** bench limpo
  **335/116/80** a 500/2000/5000 (load 2.4, piso 3.9x — ver
  "Desempenho"), `docs/demo.gif` regenerado (9000 ticks, 29 frames,
  **2.25 MB**; README não cita tamanho do GIF, nada a atualizar),
  painel fit verde (`test_panel`). Segunda amostra limpa: **269/121/62**
  (faixa 2 runs, load 2–3: 269–335 / 116–121 / 62–80 → tabela do README
  atualizada). Honestidade de publicação no `/cp`: README principal
  corrigido (genoma 427 / 28 inputs, 73 testes, bench etapa 9, roadmap
  1–9, descrição do viewer) + nota **Reproducibility** no README top e
  nos 4 experimentos (números vieram do snapshot pré-etapa-9, genoma
  19-in/283; rerun em HEAD repete o protocolo mas muda trajetória;
  re-baseline em HEAD = tarefa pós-commit opcional). **Decisão de
  commit: 1 único** (etapa 8 + 9) — dividir exigiria reconstruir
  estados de arquivo nunca versionados (`record.py` já com F3,
  `config.py`/`run.py` misturados), fabricando histórico. Falta só:
  julgamento visual do usuário (screenshots `/tmp/opencode/
  f4_selected.png` e `f4_overview.png`).
- **Etapa 8 concluída (2026-10-08; entra no commit único com a etapa
  9 antes do push):** `--config` pronto
  (`load_config(path, base)` em `config.py` + flag em `run.py`; campo
  desconhecido → `ValueError`, `--config` com `--load` → erro de CLI;
  `tests/test_config.py`). Experimento 1 (`experiments/food_supply/`)
  rodou o sweep completo 6 configs × 3 seeds × 5000 ticks (figuras +
  README com números reais; `data/` em `.gitignore`). **GATE decidido pelo
  usuário: `max_age` 5000 → 2000** (ver "Balanceamento"). Helpers em
  `experiments/_common.py` (cache por npz com staleness, bandas min-max
  entre seeds). **Fase C concluída:** 3 experimentos novos, cada um com
  `run.py` + `figs/` + `README.md` com números reais —
  `mutation_diversity` (trait_mutation_std 0.05/0.10/0.30: genome_dist
  final 3.34/7.60/12.77, monótono, pop/energia estáveis, curvas em
  degraus nos pulsos de virada), `predation` (resultado negativo
  honesto: 0/3/2 mortes por predação por run; diet_mean ≤ 0.0093;
  escassez de comida dobra a dieta mas a mordida `bite_rate*diet` é
  pequena demais para pagar — armadilha/ratchet documentada),
  `lineage` (~5600 nascimentos em pulsos, 49–56 famílias sobreviventes
  de 500 fundadores, top-10 = 74–77% da pop, profundidade mediana 6–7,
  overflow 0). **Fase D concluída:** `primordia/render/record.py`
  (`WorldRecorder` + `assemble_gif`: captura read-only em superfícies
  offscreen com SDL dummy, PNGs numerados em `DIR/frames/`, GIF via
  Pillow com downscale ≤960 px/paleta 256), CLI `--record DIR` +
  `--record-every K` (exige `--headless`; tick 0 sempre capturado),
  `tests/test_record.py` (read-only+determinismo e GIF completo) —
  `pytest -q` **56 passed**. `docs/demo.gif` gerado: 9000 ticks seed 42,
  29 frames, 3.00 MB (orçamento <4 MB; frames vivem em `docs/frames/`,
  gitignored). Evidência: `frame_00001` de `docs/` é bit-idêntico a
  re-render tick 320 em processo novo. **Fase E concluída:** `README.md`
  (EN: GIF topo, conceito, install `pip install -e ".[render,dev]"`,
  4 descobertas com números reais e figuras, bench, reprodução),
  `LICENSE` (MIT, copyright `H1lbert-kt` 2026), `pyproject.toml`
  (+`pillow>=10` no extra `render`; `[tool.setuptools] packages`
  explícitos — auto-discovery falhava no layout flat), `gh repo edit`
  (descrição nova + 8 topics: simulation, artificial-life,
  neural-networks, evolution, numba, numpy, pygame,
  agent-based-modeling). **Etapa 8 pronta para commit (`/cp`).**
- **Etapa 7 de 8 concluída**: salvar/carregar + árvore genealógica —
  `primordia/io.py` (`save_world`/`load_world`, npz **formato v1** com
  `format_version`; versão desconhecida → `ValueError`; salva só o estado
  de verdade — scratch/cells ficam de fora, `rebuild_counts` roda no load),
  `run.py --save PATH` (headless e janela) e `--load PATH` (ignora
  `--seed/--pop`, imprime tick/alive/births do arquivo). Genealogia:
  `creature_id (N,) int32` monotônico (`next_id`), **`parent_id` guarda o
  id do pai, não o slot** (slot é reciclado e corromperia a linhagem;
  `-1` = fundador), log de nascimentos `genealogy (L,3) int32` =
  `(tick, child_id, parent_id)` com `Config.genealogy_capacity=100_000` e
  contador `genealogy_overflow` (não cresce arrays); painel exibe
  `id/parent`. RNG: `genome.get/set_numba_rng_state` via
  `numba._helperlib.rnd_get_np_state_ptr` (API **privada**, verificada em
  numba 0.68, é a da suite do numba; guard em `tests/test_io.py`).
  Testes novos em `tests/test_io.py` (**54 no total** com
  `tests/test_config.py`; guard importa `primordia.io`).
- **Continuação idêntica provada na CLI:** 2500 ticks contínuos ==
  2000 + `--save` + `--load` + 500 — arrays, contadores e log bit-a-bit
  (seed 42, N=2000); em-processo no teste `test_continuation_identical`
  (sequencial + restore explícito do stream salvo entre as pernas).
- **Desempenho da etapa 7:** profile A/B alternado (clone do `a148ecb` em
  `/tmp`): `reproduce` 16 → **23 µs/tick** (+7 µs ≈ 0.4% do tick a N=2000;
  custo = atribuição de id + linha de log, **zero draws de RNG novos**).
  Bench sob load 9–12 (máquina saturada): 263–279/108–172/61–70 a
  500/2000/5000; A/B alternado oscila na mesma faixa nos dois lados —
  ruído domina, sem regressão atribuível. Piso 30 @N=2000 cumprido com
  3–6x. Re-medir o bench quando a máquina estiver com load <3.
- **Etapa 8 (config + max_age, 2026-10-08):** bench sob load 5–6 (dois
  runs seguidos): **259–344 / 85–93 / 58–64** a 500/2000/5000 — piso 30
  @N=2000 cumprido com ~3x; nenhum código do tick mudou (só default de
  `Config.max_age`, caminho quente idêntico), variação é ruído de carga.
  Re-medir com load <3.
- **Stats (etapa 6, ainda válido):** custo de `record` ~0.5 ms/tick;
  **nunca voltar `@` no `genome_dist`** (bug OpenBLAS: 35 ticks/s, usar
  `np.einsum` sem `optimize`); runs longos com `--stats` custam ~40% de
  ticks/s (medição antiga, load 3–10).
- **Balanceamento (GATE da etapa 8, resolvido 2026-10-08):** o sweep do
  `experiments/food_supply/` (6 configs: `food_growth_rate {0.01,0.05,0.10}`
  × `metabolic_cost {0.05,0.15}`, 3 seeds, 5000 ticks) mostrou que esses
  dois knobs **só mudam a inclinação** do crescimento de energia
  (~1.43–1.87 e/tick), nunca a estrutura: pop sempre no cap, predação 0,
  mortalidade real ~100 fomes/run. Causa raiz: com pop no cap,
  `_reproduce` (`genome.py`) pula o pai sem gastar energia quando não há
  slot livre → único sumidouro é metabolismo+movimento < ingestão.
  **Decisão do usuário: `max_age` 5000 → 2000** (default novo). Validação
  (seeds 42/7, 5000 ticks): energia agora limitada em ciclo ~1000–4000
  (final 3381/3029, era 7669/8397 subindo sem parar), ~4400 mortes por
  idade/run com nascimentos em pulsos ao longo do run (eram ~391 só no
  cliff final; free-list LIFO sincroniza coortes), pop segue estável em
  2000, `diet_mean` 0.001–0.005 (predação ainda inerte — é a pergunta do
  experimento `predation`). `pytest -q` (54) verde com o default novo.
- Predação/dieta ativas prontas; experimentos/README/GIFs (etapa 8).
- `pytest -q` (92 passed), `python -m primordia.bench [--profile]`,
  `run.py --headless --ticks N [--stats out.npz] [--save w.npz]`,
  `run.py --load w.npz --headless --ticks N`,
  `python plot.py out.npz --out figs/`, `run.py --seed 42` (janela),
  `run.py --config overrides.json` e
  `run.py --headless --ticks 9000 --record docs --record-every 360`
  (frames + GIF demo) funcionam. Plot da árvore genealógica
  ficou fora do catálogo aprovado da etapa 8 (lineage usa o log com
  métricas agregadas).
- **Planos das etapas** ficam em `.opencode/plans/*.md` (não versionados).
- Ambiente em `.venv` (Python **3.14.7**): numpy 2.5.3, numba 0.68.0,
  pygame-ce 2.5.8, matplotlib 3.11.2, Pillow 12.3.0, pytest 9.1.1. Verificado: `@njit` +
  `prange` compilam e `import pygame` funciona.
- **Não há ruff/mypy/black instalados.** Não instale nada novo sem perguntar.
- Use o venv: `source .venv/bin/activate` ou `.venv/bin/python ...`.
- Branch `master`; remote `origin` = `https://github.com/H1lbert-kt/primordia.git`.

## Ambiente (restrições reais, não negociáveis)
- CPU i5-8265U (4 cores / 8 threads), ~19 GB RAM, **sem GPU dedicada, sem CUDA**.
- Python 3.11+ (venv atual: 3.14.7), NumPy, Numba, **pygame-ce**, Matplotlib, pytest.
- **pygame-ce, não pygame.** Instale com `pip install pygame-ce`; o nome de
  import continua `import pygame`. **Nunca** instale o pacote `pygame`
  original junto (os dois conflitam). Se aparecer, rode
  `pip uninstall pygame pygame-ce -y && pip install pygame-ce`.
- Pode usar recursos exclusivos do pygame-ce (ex.: `pygame.FRect`,
  `pygame.Window`), mas confira na documentação do pygame-ce antes de usar
  uma API, pois ela difere do pygame original em alguns pontos.
- **Proibido** PyTorch, TensorFlow, JAX, CuPy ou qualquer dependência de GPU.
- Não adicione nenhuma dependência nova sem perguntar antes e justificar.
- Não rode treinos/simulações longas (>2 min) sem avisar o usuário.

## Princípios inegociáveis
1. **Sem fitness explícito.** Reprodução = energia acima de um limiar. Morte =
   energia <= 0, idade máxima ou predação. Nunca premie "inteligência",
   distância percorrida ou comportamento específico.
2. **Núcleo headless.** O pacote `primordia/sim` não pode importar pygame nem
   matplotlib. A renderização lê o estado; nunca o modifica.
3. **Determinismo.** Mesma `--seed` => mesma simulação, bit a bit, na mesma máquina.
4. **Dados em arrays (SoA).** Uma criatura é um índice, nunca um objeto.
5. **Parâmetros em config.** Nenhum número mágico de balanceamento no código.

## Estrutura (alvo)
```
primordia/
  config.py      # dataclass Config (todos os parâmetros de balanceamento)
  world.py       # estado do mundo: arrays, comida, grid espacial
  genome.py      # codificação, mutação, crossover/reprodução
  brain.py       # MLP vetorizada em Numba
  sensors.py     # raios de visão, cheiro, energia interna
  step.py        # tick da simulação (única função que avança o mundo)
  stats.py       # métricas, histórico, diversidade, espécies
  io.py          # salvar/carregar mundo (npz) com versão de formato
  render/        # pygame: câmera, zoom, painel do cérebro (nunca importado pelo núcleo)
tests/
experiments/     # um script + um README curto por experimento
run.py           # CLI: --seed --headless --ticks --load --save --stats
                 #      --config --screenshot --select (janela)
```

## Modelo de dados
- Capacidade fixa pré-alocada (`max_creatures`), com máscara `alive` (bool).
  Mortos são reciclados via free-list; **nunca** `append`/`delete` em arrays no loop.
- Campos mínimos por criatura: `pos (N,2)`, `vel (N,2)`, `angle`, `energy`,
  `age`, `genome (N,G)`, `species_id`, `parent_id` (id do pai), `creature_id`,
  `alive`.
- Use `float32` para estado e pesos, `int32` para ids/contadores. Documente
  o dtype de cada array em `world.py`.
- Cérebro: MLP pequena (entrada ~ raios + sensores internos, 1 camada oculta
  de 8–16 neurônios, saídas: aceleração, giro, comer/atacar). Meta: 100–500
  parâmetros por criatura. O genoma contém pesos do cérebro **e** traços do
  corpo (velocidade máx., tamanho, alcance de visão, dieta).

## Desempenho (regras duras)
- **Proibido loop Python sobre criaturas.** Toda lógica por criatura vive em
  funções `@njit(cache=True)`, usando `prange` quando seguro, ou em NumPy vetorizado.
- Detecção de vizinhança com **grade espacial** (spatial hash), nunca O(N²).
- Sem alocação dentro do tick: reutilize buffers pré-alocados.
- Funções Numba: tipos homogêneos, sem objetos Python, sem listas de dicts.
- Meta: **2000 criaturas a >= 30 ticks/s** em headless. Se uma mudança fizer
  isso cair >15%, investigue e reporte antes de seguir.
- Após toda etapa que mexa no tick, rode `python -m primordia.bench` e mostre
  ticks/s com 500, 2000 e 5000 criaturas. Não declare "pronto" sem esse número.
- Meça antes de otimizar (use `time.perf_counter`, aqueça o JIT antes de medir).
- Baseline da etapa 1 (última medição 2026-10-06, i5-8265U):
  11240/7874/3999 ticks/s a 500/2000/5000 criaturas (~2x de variação entre
  execuções conforme a carga da máquina; compare sempre na mesma sessão).
- Baseline da etapa 2 (sensores + cérebro ativos, medição 2026-10-07, i5-8265U):
  ~2500/830/300 ticks/s a 500/2000/5000 criaturas (mesma variância de carga;
  1 outlier de 586 a N=2000 sob load 1.8). Gargalo medido com `--profile`
  a N=2000: `perceive` 64% (amostragem de raios, leituras aleatórias em
  `food`/`cell_counts`), `think` 25% (12 `tanh` por criatura). O piso de
  30 ticks/s é cumprido com 27x de margem.
- **Etapa 3 (2026-10-07): sem regressão medida.** O ticks/s oscila 2–4x
  *entre processos* nos dois palcos (ruído de threads Numba/térmico) — número
  isolado não serve para comparar etapas. Protocolo: A/B alternado na mesma
  sessão (clone do HEAD anterior em `/tmp` + `--profile`). Resultado:
  `sum(phases)` a N=2000 = 2691/2292 µs (etapa 3) vs 3297/2375 µs (etapa 2);
  `reproduce` = 8.5 µs (0.3%). Pior caso observado da etapa 3: ~110 ticks/s
  a N=2000 (piso 30 cumprido com ~4x). Em 5000 ticks com população crescendo
  (500→2000 cap): ~284 ticks/s, dinâmica em `--profile`/testes de balanço.
- **Etapa 4 (janela pygame, 2026-10-07):** N=2000, 300 ticks, fps médio
  oscila **21–47** conforme a carga da máquina (o mesmo box roda
  opencode/Brave/gnome-shell; rode `uptime` junto da medição e compare A/B
  alternado na mesma sessão). Pipeline isolado `SDL_VIDEODRIVER=dummy`:
  38–47 fps; X real: 26–43 fps (load 2–6). Sob load 8–12 cai a ~24–27 —
  o limitante é a carga do sistema, não o draw (por frame: draw_food ~5 ms,
  panel ~2 ms, stamp <1 ms). Cuidado ao ler buckets: acumulam **segundos**;
  `s/frame` sem ×1000 mostra 0.04 quando o real é **40 ms**.
- **Etapa 5 (bite/CSR, 2026-10-07):** A/B alternado com clone do `d745520`
  em `/tmp` (bench oscilou 137–324 ticks/s nos dois lados sob load 6–10 —
  ruído domina o bench; confie no profile por fase). Custo novo medido em
  processo quente: `rebuild_counts` ~91–105 µs (era ~22 µs; prefix com
  `reshape(-1)` derrubou de 230 para 91; restante = zeração CSR + passes),
  `bite` ~21–29 µs (no-op barato com diet 0). Bench nesta sessão (load
  6–8): **1161/382/186 ticks/s** a 500/2000/5000 (piso 30 cumprido com 6x;
  variação entre execuções até 2–3x conforme carga).
- **Etapa 6 (stats, 2026-10-07):** bench **545/527/114 ticks/s** a
  500/2000/5000 (load ~3; o stats não mexe no tick). Custos do `record`
  medidos em loop quente: ~0.46–0.62 ms/tick (traits mean/std ~170 µs,
  energia/idade ~30 µs, dist amostrada amortizada). Efeito OpenBLAS
  (ver Estado atual): nunca comparar `@` vs `einsum` só isolado — medir o
  loop completo. Runs longos com `--stats`: ver "Desempenho do stats".
- **Etapa 7 (io + genealogia, 2026-10-08):** bench sob **load 9–12**:
  263–279/108–172/61–70 a 500/2000/5000; A/B alternado contra o clone do
  `a148ecb` oscila na mesma faixa nos dois lados (ruído domina o bench
  total; confie no profile por fase). Profile por fase: `reproduce`
  16 → **23 µs** (id + log, sem draws novos) — único custo da etapa no
  tick. Piso 30 @N=2000 cumprido com 3–6x. Re-medir com load <3.
- **Etapa 9 F4 (2026-10-09):** bench sob **load 6–18** (runc/containers
  saturando o box durante a medição): 44–49/21–22/17–18 a 500/2000/5000 —
  **não conclusivo** (N=2000 abaixo do piso, mas a carga explicaria).
  Profile por fase sob carga: `sum(phases)` 38.7 ms (~26 t/s), fases
  "planas" (perceive 26% / think 25% / apply 21% / integrate 26%) —
  overhead de agendamento de threads domina kernels pequenos sob carga,
  shares não comparáveis com runs limpos. Custos esperados do F4 no tick
  (estimativa FLOPs): think +~55% (28 in + W_rec·prev) ≈ +13% do tick.
  **Re-medir com load <3 antes de declarar o piso cumprido**; se <30,
  investigar (candidatos: cópia hidden_prev→hidden_buf em think,
  28 inputs em perceive).
- **Etapa 9 F4 — re-medição limpa (2026-10-09):** janela silenciosa
  (load 2.4 antes do bench; o processo "Main"/container de 115% CPU
  terminou): **335/116/80 ticks/s** a 500/2000/5000 — piso 30 @N=2000
  cumprido com **3.9x**, faixa igual ou melhor que a etapa 8 (85–93 sob
  load 5–6) → **sem regressão atribuível ao F4**. Demo GIF regenerado na
  mesma janela: 9000 ticks em 83.1 s (108 t/s) → 29 frames, **2.25 MB**
  (era 3.00; orçamento <4 MB; backup do antigo em
  `/tmp/opencode/demo_stage8_3.00mb.gif`).
- O Numba usa todas as 8 threads lógicas: rode benchmark e simulação em
  **sequência**. Dois processos Numba paralelos se pisoteiam e o ticks/s cai
  ~20x (medido em 2026-10-06: 19 ticks/s em paralelo vs ~8000 em sequência).

## Determinismo
- Um único `numpy.random.Generator` (`default_rng(seed)`) no mundo para
  desenhos em Python (spawn/inicialização).
- O gerador do Numba é separado do NumPy: se usar `np.random` dentro de
  `@njit`, semeie-o explicitamente com `np.random.seed(...)` dentro de uma
  função `@njit` (é o caso de `genome.seed_numba_rng`).
- Esse stream do Numba é **global por processo**: dois mundos avançando
  intercalados compartilham desenhos e divergem. Com reprodução ativa,
  rode mundos em **sequência** (mundo A completo, depois B) — CLI e bench
  já são sequenciais; testes seguem a mesma regra.
- Save/load (`primordia.io`) captura e restaura **ambos** os streams
  (`rng.bit_generator.state` + `get/set_numba_rng_state`), então
  `load(save(w))` continua bit-a-bit; salvar exige que o mundo seja o
  stream ativo (mesma regra sequencial de cima).
- Dentro de `prange`, **não** gere números aleatórios. Pré-gere os arrays de
  ruído fora do bloco paralelo e passe-os como argumento.
- Nunca itere sobre `set`/`dict` onde a ordem afete o resultado.
- Todo bug de determinismo tem prioridade sobre novas funcionalidades.
- Teste obrigatório: duas execuções com a mesma seed e 500 ticks produzem
  arrays idênticos.

## Balanceamento
Simulações assim falham por balanceamento, não por bug. Portanto:
- Todos os parâmetros (custo de movimento, custo de metabolismo, energia da
  comida, taxa de reaparecimento, limiar de reprodução, taxa de mutação,
  energia ganha por predação) ficam em `Config`.
- Mantenha `stats.py` registrando por tick: população, energia média, idade
  média, nascimentos, mortes por causa (fome, idade, predação), diversidade.
- Se a população extingue ou explode, **não "conserte" com hacks** (clamp
  mágico, spawn de emergência). Reporte as métricas e proponha ajuste de
  parâmetros em `Config`, e deixe o usuário decidir.
- Evite criar dinâmicas que dão vantagem direta a um comportamento
  específico (isso viola o princípio 1).

## Fluxo de trabalho do agente
1. **Planeje antes de codar** em tarefas grandes: descreva abordagem, arquivos
   afetados e riscos; espere aprovação.
2. **Passos pequenos e verificáveis.** Uma funcionalidade por vez. Não
   refatore código fora do escopo do pedido.
3. **Teste primeiro ou junto.** Todo módulo novo vem com testes em `tests/`.
   Rode `pytest -q` e mostre o resultado.
4. **Mostre evidência**, não afirmações: saída de teste, ticks/s, gráfico
   ou GIF. Se não executou, diga que não executou.
5. **Não invente.** Se não souber a API de uma biblioteca, leia a
   documentação ou o código instalado em vez de chutar.
6. **Peça confirmação** antes de: apagar arquivos, mudar o formato de
   salvamento, adicionar dependências, mudar a estrutura de pastas.
7. Ao terminar, resuma em poucas linhas: o que mudou, o que foi testado,
   números de desempenho, pendências conhecidas.

## Comandos
```bash
python run.py --seed 42 --headless --ticks 10000   # simulação sem janela
python run.py --seed 42 --headless --ticks 5000 --stats s42.npz
python plot.py s42.npz --out figs/                 # 4 PNGs (matplotlib)
python run.py --seed 42                            # com visualização
python run.py --seed 42 --ticks 300 --select 1 \
    --screenshot out.png                           # janela com evidência
python run.py --seed 42 --headless --ticks 5000 \
    --save saves/w5k.npz                           # gravar mundo (etapa 7)
python run.py --load saves/w5k.npz --headless --ticks 500   # retomar
pytest -q                                          # testes
python -m primordia.bench                          # benchmark de ticks/s
```
Valores a partir do root do repo, com o venv ativo. `pytest` de teste único:
`.venv/bin/pytest tests/teste_x.py -q`. Rodar o benchmark e colar os números
antes de dizer que uma mudança de tick não regrediu desempenho.

## Renderização (pygame-ce)
- Nunca desenhe criatura por criatura com chamadas `pygame.draw` em loop
  Python para milhares de entidades. Use `surfarray`/`pygame.surfarray` com
  arrays NumPy, ou desenhe só as criaturas visíveis na câmera, ou use
  sprites pré-renderizados por tamanho/cor.
- A renderização deve rodar em taxa independente da simulação (ex.: simular
  vários ticks por frame) e permitir pausar e acelerar.
- Câmera com zoom e arrasto; painel lateral mostrando o cérebro e os
  sensores da criatura selecionada.
- Para gravar GIF/vídeo, salve frames em disco a partir do estado do mundo,
  nunca dentro do loop de simulação.

## Testes (o que sempre deve ser coberto)
- Conservação/consistência: nenhuma criatura viva com `energy <= 0`; nenhuma
  posição fora do mundo (trata bordas: wrap ou parede, conforme Config).
- Determinismo (ver acima).
- Mutação: taxa e magnitude respeitam a Config; genoma mantém o shape.
- Reprodução: filho herda traços do pai com mutação; energia é dividida
  (sem criar energia do nada).
- Salvar/carregar: `load(save(world))` continua a simulação de forma idêntica.
- Renderização: só testes de guarda/smoke (núcleo sem pygame; janela dummy
  idêntica ao headless). Não há CI gráfica; o núcleo sim, sim.

## Estilo de código
- Type hints em funções públicas; docstring curta dizendo *o quê e por quê*.
- Funções pequenas, nomes claros, sem abreviações crípticas.
- `ruff` para lint/format quando existir; hoje não está instalado, e não
  adicione o pacote sem perguntar. Não brigue com ele.
- Sem `print` de debug esquecido; use `logging` ou flags do CLI.
- Sem código morto, sem comentários que apenas repetem o código.

## Git
- Commits pequenos, um por funcionalidade, no formato convencional:
  `feat: ...`, `fix: ...`, `perf: ...`, `test: ...`, `docs: ...`, `refactor: ...`.
- Nunca faça commit de `saves/`, `.venv/`, `__pycache__/`, GIFs/vídeos
  grandes (>5 MB) ou resultados brutos de experimentos longos.
- Antes de qualquer mudança grande, sugira ao usuário fazer um commit para
  poder reverter.

## Experimentos e documentação
- Cada experimento em `experiments/<nome>/` com: script reproduzível
  (`--seed`), saída de dados, gráficos, e um `README.md` curto com pergunta,
  método, resultado e limitações.
- Reporte resultados com honestidade: múltiplas seeds, média e variação.
  Não afirme "emergiu X" sem mostrar evidência (métrica, trajetória,
  GIF) e sem checar se não é artefato de um parâmetro.
- README principal: GIF de demo no topo, conceito em uma frase, instalação em
  um comando, resultados com gráficos, seção "o que descobri", e como
  reproduzir. Mantenha em inglês; versão `README.pt-BR.md` opcional.

## Definição de pronto (para cada etapa)
- [ ] Código roda headless e com `--seed` determinístico
- [ ] `pytest -q` passa
- [ ] Benchmark executado e meta de desempenho respeitada (ou desvio explicado)
- [ ] Sem loops Python sobre criaturas, sem dependência nova não aprovada
- [ ] Commit feito com mensagem convencional

## Roadmap (ordem das etapas)
1. Mundo + comida (headless, benchmark)
2. Cérebro MLP vetorizado + sensores em raios
3. Genoma, mutação, reprodução por energia
4. Renderização Pygame (câmera, zoom, painel do cérebro)
5. Predação como traço de dieta evoluível
6. Estatísticas e gráficos
7. Salvar/carregar e árvore genealógica
8. Experimentos, GIFs, README e publicação
9. "Mundo que ensina" (patches, sentidos ricos, Elman, viewer) ✓
9.5. Balanço e pressão seletiva (economia, turn_cost, zoom) ✓

Roadmap "mundo real" aprovado (`.opencode/plans/roadmap-mundo-real.md`):
10 ambiente (terreno/estações/luz) ✓ → 11 biologia (sexualidade/senescência/carniça) ✓
→ 12 ecologia (ciclos tróficos) ✓ → 13 canal ✓ → 14 difusão ✓ →
15 tag/cor ✓ → 16 doação + `experiments/proto_culture/` ✓ →
17 civilização (madeira/fogo/casas) + `experiments/civilization/` ✓ →
18 balanceamento dos defaults ✓ (próxima: daemon+SQLite → território →
Oja, a decidir com o usuário).

Trabalhe **apenas na etapa que o usuário indicar**.
