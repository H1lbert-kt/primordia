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

## Estado atual do repositório (verificado em 2026-10-07)
- **Etapa 2 de 8 concluída**: `primordia/` ganhou `sensors.py` (raios +
  cheiro + sensores internos) e `brain.py` (MLP `think` + `apply_actions`);
  `config.py`/`world.py`/`step.py`/`bench.py` expandidos; testes novos em
  `tests/test_brain.py` e `tests/test_sensors.py` (13 no total).
  Não há README, CI, `opencode.json` nem renderização (etapa 4); genoma/muta-
  ção/reprodução são etapa 3.
- `pytest -q` (13 passed), `python -m primordia.bench`,
  `python -m primordia.bench --profile` e
  `run.py --headless --ticks N` funcionam. `run.py` sem `--headless` e
  `--load saves/...` só passam a funcionar nas etapas 4 e 7.
- **Planos das etapas** ficam em `.opencode/plans/*.md` (não versionados).
- Ambiente em `.venv` (Python **3.14.7**): numpy 2.5.3, numba 0.68.0,
  pygame-ce 2.5.8, matplotlib 3.11.2, pytest 9.1.1. Verificado: `@njit` +
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

## Estrutura (alvo; nada disso existe ainda)
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
run.py           # CLI: --seed --headless --ticks --load --config
```

## Modelo de dados
- Capacidade fixa pré-alocada (`max_creatures`), com máscara `alive` (bool).
  Mortos são reciclados via free-list; **nunca** `append`/`delete` em arrays no loop.
- Campos mínimos por criatura: `pos (N,2)`, `vel (N,2)`, `angle`, `energy`,
  `age`, `genome (N,G)`, `species_id`, `parent_id`, `alive`.
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
- O Numba usa todas as 8 threads lógicas: rode benchmark e simulação em
  **sequência**. Dois processos Numba paralelos se pisoteiam e o ticks/s cai
  ~20x (medido em 2026-10-06: 19 ticks/s em paralelo vs ~8000 em sequência).

## Determinismo
- Um único `numpy.random.Generator` (`default_rng(seed)`) no mundo.
- O gerador do Numba é separado do NumPy: se usar `np.random` dentro de
  `@njit`, semeie-o explicitamente com `np.random.seed(...)` dentro de uma
  função `@njit`.
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
python run.py --seed 42                            # com visualização
python run.py --load saves/world.npz               # retomar mundo
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
- Renderização não é testada em CI; o núcleo sim, sim.

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
