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

## Estado atual do repositório (verificado em 2026-10-09)
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
- `pytest -q` (73 passed), `python -m primordia.bench [--profile]`,
  `run.py --headless --ticks N [--stats out.npz] [--save w.npz]`,
  `run.py --load w.npz --headless --ticks N`,
  `python plot.py out.npz --out figs/`, `run.py --seed 42` (janela),
  `run.py --config overrides.json` e
  `run.py --headless --ticks 9000 --record docs --record-every 320`
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

Trabalhe **apenas na etapa que o usuário indicar**.
