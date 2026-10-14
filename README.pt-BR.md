# Primordia

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![tests: 311 passed](https://img.shields.io/badge/tests-311%20passed-brightgreen.svg)](tests/)

![demo](docs/demo.gif)

**Um ecossistema 2D onde criaturas neurais evoluem por seleção natural — não
existe função de fitness em lugar nenhum.** Criaturas que sobrevivem e
acumulam energia se dividem em duas; as que morrem de fome, envelhecem demais
ou são mordidas, morrem. Predação, bandos e fuga não são programados — ou
emergem da seleção sobre os traços de corpo/cérebro evoluídos, ou não
emergem. Este repositório é uma tentativa honesta de descobrir qual dos dois,
com experimentos reproduzíveis e resultados negativos publicados.

**Read in English:** [README.md](README.md)

```bash
pip install -e ".[render,dev]"
python run.py --seed 42                # visualizador com janela
python run.py --seed 42 --headless --ticks 5000 --stats s42.npz
pytest -q
```

## Sumário

- [Como funciona](#como-funciona)
- [Inspirations](#inspirations)
- [O que eu descobri](#o-que-eu-descobri)
- [Dificuldades e problemas abertos](#dificuldades-e-problemas-abertos)
- [Visualizador](#visualizador)
- [O mundo que roda sozinho](#o-mundo-que-roda-sozinho)
- [Desempenho](#desempenho)
- [Reproduza tudo](#reproduza-tudo)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Roadmap (25 etapas)](#roadmap-25-etapas)
- [Licença](#licença)

## Como funciona

- **Sem fitness, sem objetivos.** Energia é a única moeda: coma comida (ou
  carne), pague os custos de metabolismo e movimento, e se divida quando
  cruzar o `reproduce_threshold`. Morte = energia ≤ 0, `max_age`, ou a mordida
  de um predador. Nada no tick jamais premia "inteligência", distância
  percorrida ou qualquer comportamento em particular.
- **Corpo e cérebro evoluíveis.** Cada criatura é um genoma de **454
  valores**: uma rede de **38 entradas** (pico de comida em 7 raios, criaturas,
  presa e carniça, antenas esq/dir de comida, estado interno, luz ambiente,
  três leituras do campo de feromônio, a similaridade de cor dos vizinhos na
  janela de cheiro, o cheiro de madeira dessa mesma janela, duas cicatrizes
  de mordida — estresse acumulado e rancor que decai com o tempo em relação à
  cor do último atacante — e, desde a etapa 25, uma leitura assinada do
  banco: + se você está sobre a própria casa com depósito, − sobre uma casa
  alheia) → **8 unidades ocultas** com recorrência de Elman → 8 saídas:
  acelerar, girar, comer/atacar/sacar/roubar, emitir sinal, doar energia,
  coletar madeira, construir uma fogueira, construir uma casa — mais **6
  traços corporais**: velocidade máx., tamanho, alcance de visão, *dieta*
  (0 = herbívoro; a capacidade de predação é ela própria um traço que precisa
  evoluir), uma cor de linhagem herdável — neutra por construção, mostrada
  como um anel sob o sprite — e a taxa de aprendizado η da regra de
  plasticidade. Entradas informativas (estresse, rancor, banco) são
  **leituras do que o mundo já contém, nunca prêmios**; o canal afetivo está
  desligado por padrão e o experimento que o ligou reduziu a população pela
  metade (etapa 24, abaixo).
- **Sexo e envelhecimento, não só clonagem.** A reprodução é sexual: um
  iniciador acima do limiar procura um par por perto, cada pai paga metade da
  energia do filho, e o genoma é um cruzamento gene a gene mais mutação. Corpos
  velhos se desgastam (`w = exp(senescence_rate * age)`): o metabolismo cresce
  enquanto velocidade e visão encolhem. Sem par → sem nascimento: mobilidade e
  agregação são selecionadas sem serem programadas.
- **Um mundo que revida.** Fertilidade do terreno e lagos moldam onde a comida
  cresce, um sol define o ciclo dia/noite (a visão encolhe à noite) e as
  estações modulam o crescimento — tudo física pura vinda do `Config`, sem
  objetivos. Madeira cresce em patches; fogueiras cozinham; casas podem ser
  erguidas em entregas, protegem o dono e apodrecem se inacabadas; energia
  depositada pode ser guardada ou roubada; uma marca no chão desconta sua
  conta metabólica.
- **Núcleo headless, renderização unidirecional.** Os módulos centrais
  (`world`, `step`, `brain`, `genome`, `sensors`, `stats`, `io`) nunca
  importam pygame nem matplotlib; o visualizador apenas lê o estado.
- **Determinístico.** Mesmo `--seed` ⇒ simulação bit-idêntica na mesma
  máquina, inclusive através de save/load (`--save`/`--load` restauram os dois
  streams de RNG e o log de genealogia).
- **Orientado a dados.** Arrays SoA de capacidade fixa com free-list — nenhum
  objeto Python por criatura, nenhuma alocação dentro do tick, vizinhança por
  hash espacial, tudo quente em Numba `@njit`.

## Inspirations

O Primordia se situa numa linhagem de projetos de vida artificial que fazem a
mesma pergunta — *quanto pode emergir de muito pouco?*

- **Conway's Game of Life (1970)** — a demonstração fundadora de que regras
  locais, fixas e burras produzem planadores, canhões e computação sem ninguém
  no comando. A regra do Primordia tem a mesma forma: uma criatura apenas
  come, paga e se divide; predação, bandos e fronteiras não estão escritas em
  lugar nenhum.
- **Tierra (Thomas Ray, 1991) e Avida** — organismos digitais de vida livre
  que competem e se reproduzem no código, estudados *sem* uma função de
  fitness alimentada à mão; o Avida em particular mostrou como *medir* a
  evolução em vez de prescrevê-la. Esse é o método aqui: rode o mundo, depois
  meça.
- **Polyworld (Larry Sims, 1996)** — um ecossistema 2D onde cérebros e corpos
  são codificados no genoma e desenhados como um mapa vivo: um direto
  ancestral do visualizador do Primordia.
- **NEAT e redes neurais evoluíveis (Stanley & Miikkulainen, 2002)** — o
  hábito de colocar os pesos de conexão no genoma e deixar a seleção moldar o
  comportamento. O Primordia mantém uma topologia fixa e deliberadamente
  pequena (o orçamento de 500 parâmetros) em vez de evoluir a estrutura, de
  modo que cada sentido novo precisa se pagar dentro de um orçamento apertado.
- **The Blind Watchmaker (Dawkins, 1986)** — os biomorfos mostrando que a
  seleção, e não o design, explica a aparência de propósito. "Sem fitness, sem
  objetivos" é essa ideia levada literalmente.

O que este projeto acrescenta ao sonho antigo é a disciplina de engenharia ao
redor dele: determinismo bit-exato, saves versionados, experimentos
pré-registrados com falhas publicadas, e um piso de desempenho que roda no CPU
de um notebook.

## O que eu descobri

Resultados acumulados na suíte de experimentos
([`experiments/`](experiments/)) — etapas 8 a 25, na maioria 3 seeds ×
5 000–10 000 ticks cada. Método completo e limitações no README de cada
experimento.

> **Nota de reprodutibilidade.** As tabelas e figuras abaixo foram produzidas
> na árvore pré-etapa-9 (cérebros feedforward de 19 entradas e 12 ocultos,
> campo de comida uniforme, a economia antiga de comida). A etapa 9 mudou o
> layout do cérebro (28 entradas, 10 unidades ocultas, recorrência de Elman),
> a etapa 9.5 recalibrou a economia, a etapa 10 adicionou uma 29ª entrada
> (luz ambiente) mais terreno/dia-noite/estações, a etapa 11 passou à
> reprodução sexual e adicionou senescência e carniça (30 entradas,
> `FORMAT_VERSION` 4), a etapa 12 alargou o log de genealogia com o par
> reprodutivo e adicionou as colunas de espécies/barreira
> (`FORMAT_VERSION` 5), a etapa 13 ligou o canal comum de sinais (33 entradas,
> uma 4ª saída do cérebro, `FORMAT_VERSION` 6), a etapa 14 fez esse canal
> difundir — uma passada de Jacobi espalha o campo em trilhas (só Config,
> `FORMAT_VERSION` continua 6), a etapa 15 adicionou uma cor de linhagem
> neutra (34 entradas, um 5º traço corporativo, `FORMAT_VERSION` 7), a etapa
> 16 abriu um canal de doação (uma 5ª saída do cérebro presenteando o vizinho
> mais próximo, 9 unidades ocultas, `FORMAT_VERSION` 8), a etapa 17 adicionou
> a base de civilização (uma entrada de cheiro de madeira e três saídas para
> coletar/construir-fogo/construir-casa, `FORMAT_VERSION` 9), a etapa 18
> recalibrou *apenas* os preços padrão (fogo 10 → 5, casas precificadas para
> fora do mundo padrão em 1e6 — Config, sem mudança de tick), a etapa 19
> adicionou o daemon SQLite (só Config/CLI/armazenamento), e a etapa 21 abriu
> plasticidade de Oja vitalícia num `dw` separado (6º traço corporativo η,
> 491 parâmetros de genoma, `FORMAT_VERSION` 10 — as sinapses efetivas são
> `genome + dw`, e o genoma nunca é escrito dentro de uma vida). A
> plasticidade **por padrão está desligada**, e com ela o tick é bit-idêntico
> à lógica pré-etapa-21; mas o genoma cresceu 490 → 491 (o traço η adiciona
> um sorteio de mutação no nascimento), o que desloca o stream compartilhado
> do Numba — então mesmo reruns com `off` no HEAD atual podem produzir
> trajetórias e números diferentes dos caches das etapas 19/20. A reprodução
> bit-exata das figuras antigas ainda exige o commit antigo. As etapas 22–23
> mantiveram o layout v10 (casas incrementais e o observador de grupos
> off-tick são mudanças só de Config/scratch); a etapa 24 bumpou o layout mais
> uma vez — o par afetivo (estresse + rancor) acrescentou duas entradas (37
> in), e o orçamento forçou hidden 9 → 8, então o genoma *encolheu* 491 → 446
> (`FORMAT_VERSION` 11; os scratches afetivos em si são estado vitalício e
> ficam fora do save). A etapa 25 adicionou o banco territorial (uma entrada
> `store` assinada lendo a energia depositada sob os pés, 38 in, genoma 446 →
> 454, `FORMAT_VERSION` 12 — `struct_deposit` é salvo; o anel de marca, o
> dono da marca e o mapa de flash de roubo são scratch e reconstruídos). Os
> scripts de experimento fixam o regime antigo via `STAGE8_BASE` para que o
> *regime* seja preservado — mas reruns no HEAD atual usam a dinâmica sexual;
> as execuções assexuadas originais permanecem no cache `data/` (ignorado pelo
> git) de cada experimento (reprodução bit-exata exige o commit pré-etapa-9).

### 1. A energia cresce sem teto — e os knobs óbvios não consertam

Um sweep de 6 pontos de `food_growth_rate {0.01, 0.05, 0.10}` ×
`metabolic_cost {0.05, 0.15}` deixou a população presa no cap em todas as
variantes e a energia média subindo linearmente (~1,43–1,87 e/tick). Causa
raiz: enquanto a população fica em `max_creatures`, a reprodução bloqueada
não custa nada ao pai, então metabolismo + movimento são os únicos sumidouros
de energia e perdem para a ingestão. Os knobs só mudam a inclinação, nunca a
estrutura.

![energy](experiments/food_supply/figs/energy.png)

**Correção decidida antes de publicar:** `max_age` 5000 → 2000. A rotação
contínua por idade libera slots, reprodução e mutação seguem rodando o tempo
todo, e a energia média se estabiliza num ciclo limitado de ~1000–4000 em vez
de crescer sem limite (~4400 mortes por idade e nascimentos por run em vez de
um cliff no tick final).

**Atualização da etapa 9.5:** o knob que faltava era a *escala* da economia de
comida, não sua inclinação — `food_capacity` 100 → 3, `eat_rate` 4 → 1,
`food_growth_rate` 0.05 → 0.02, `metabolic_cost` 0.05 → 0.08. Oferta agora
equilibra demanda: em 10k ticks a população fica com média 1342–1734 e mínimos
de 185–295 (abaixo do cap de 2000), a fome é a segunda causa de morte
(11–27% das mortes) e só 1–32% dos sobreviventes ficam acima do limiar de
reprodução a qualquer momento. A seleção finalmente tem o que atacar.

**Atualização da etapa 10:** fertilidade do terreno (média ~0,43) e luz do
dia (média 0,75) cortam a oferta efetiva de comida ~60%, então
`food_growth_rate` foi recalibrado 0,02 → 0,12 (sweep 0,04–0,27, 2 seeds ×
6000 ticks). Com o ambiente ligado, 3 seeds × 10k ticks mantêm pop média
1058–1436 (mín 110–161), fome 26–37% das mortes, 1–11% dos sobreviventes
acima do limiar — e as criaturas vivem em células ~0,17–0,22 mais férteis
que a média do mapa, com a comida total acompanhando a estação
(r = 0,39–0,59) e ~5% mais comida de dia que à noite.

### 2. Mutação controla dispersão, não direção

Varrer `trait_mutation_std {0,05, 0,10, 0,30}` move a distância final de
genoma amostrado **3,34 → 7,60 → 12,77** monotonamente sem custo de
viabilidade, mas as *médias* dos traços mal se movem (±1%). Nessa janela o que
medimos é **deriva** evolutiva, não seleção direcional — e a diversidade só
cresce em degraus, nos pulsos de rotação, porque a mutação acontece no
nascimento.

![diversity](experiments/mutation_diversity/figs/diversity.png)

### 3. A predação não emergiu (resultado negativo)

Partindo de uma população inteiramente herbívora (dieta = 0 no spawn), nem a
escassez de comida nem uma recompensa de carne mais rica produziram predação
significativa: diet_mean fica ≤ 0,0093 e as mortes por predação são
**0 / 3 / 2 por run** contra ~4000 mortes por idade. A provável armadilha: o
orçamento de mordida é `bite_rate × diet`, então perto de dieta ≈ 0 a carne
não paga a própria mordida, e a dieta não cresce sem a carne pagar.
Publicado como está — um negativo honesto vale mais que um gif escolhido a
dedo.

*Atualização da etapa 9.5:* com a economia mais escassa (e um bife agora com
mais valor relativo a um orçamento de energia que encolhe), a predação ficou
pela primeira vez diferente de zero: todas as 3 seeds × 10k ticks registram
mortes por predação (1–64 por run), e o top 1% das dietas evoluídas alcança
0,17–0,22. Ainda um fio, não uma teia trófica — e a etapa 12 (abaixo)
confirma que o fio nunca virou pirâmide, mostrando ao mesmo tempo que ele
*forma* isolamento genômico.

*Atualização da etapa 10:* populações mais densas na nova economia de comida
aumentam as taxas de encontro — predação em 3 seeds × 10k vai a 1926 / 2 / 1
mortes (seed 42: essencialmente zero até o tick 7k, depois um surto de
onívoros no boom final com a pop no cap). Alta variância, ainda nenhum
predador obrigatório com dieta > 0,5 vivo em qualquer snapshot — mesmo
problema aberto, números maiores.

*Atualização da etapa 12:* medindo em vez de ajustar, a pirâmide trófica
continua plana (≥99,9% de cada run é herbívoro, 0 predadores vivos, a fome
domina as mortes), mas a pergunta de *estrutura* da etapa respondeu: os
clusters genômicos crescem 1 → ~18 em 10k ticks e emerge uma barreira
reprodutiva pequena e sustentada (fração média de acasalamentos entre
clusters 0,100 com recompensa de carne vs 0,057 com `bite_efficiency = 0`).
A predação segue com 1,5% de todas as mortes — um fio com efeito mensurável
abaixo, no isolamento, não uma teia trófica. Ver `experiments/ecology/`.

![diet](experiments/predation/figs/diet.png)

### 4. Poucas famílias dominam os sobreviventes

Com a rotação do `max_age = 2000`, ~5600 nascimentos por run chegam em
pulsos afiados (coortes sincronizadas reenchendo slots liberados em LIFO). Dos
500 fundadores, só **49–56 famílias** sobrevivem até o tick 5000, e o **top
10 responde por 74–77% da população**; profundidade mediana da linhagem 6–7
gerações.

![lineage](experiments/lineage/figs/lineage.png)

### 5. Aprendizado vitalício existe — e a seleção o precifica (etapa 21)

Sinapses agora se adaptam durante a vida pela regra local de Oja, **sem
recompensa**: o delta aprendido `dw` cavalga o genoma imutável, decai de volta
ao blueprint, e cada unidade de `|dw|` movida é cobrada da energia da
criatura (um preço fixo, `plasticity_cost=1.0` — mantido constante em toda a
etapa; qualquer mudança de preço é um experimento de sensibilidade à parte,
não misturado a estes resultados). Em 6 variantes × 3 seeds × 10k
([`experiments/plasticity/`](experiments/plasticity/)), Oja puro nunca vence
o controle (η fixo 1e-4/1e-3 termina com 832–1155 vivos vs 1053–1531; η=1e-2
**extingue 3/3 por volta de 5k** — a janela de preço fecha acima de 1e-3). O
gene η evoluível (fixado em `configs/proto_culture.toml`; o default do
`Config` é **desligado**, a linha de base que os experimentos antigos
mediram) sobrevive com **1141/1265/1461** e aprendizado engajado (`|dw|` até
2.0), ainda raro — 0,6–0,75% dos sobreviventes carregam η>0. O teste de
seleção foi rodado e diz *supressão*, não promoção: a média de η (e seu max, e
sua fração) fica **abaixo do próprio controle de deriva neutra do gene em
todas as 3 seeds** (final η̄ 2–5×10⁻⁵ com plasticidade vs 4–10×10⁻⁴ à deriva
sem uso; idem para as médias no meio do run). Pelo critério fixado antes de
ler os dados, isso é seleção negativa — o preço está sendo pago — e os
sobreviventes aprendizes são equilíbrio mutação–seleção, não prova de que a
seleção *ligou* o aprendizado. A única variante com crédito local (o
aprendizado lê a própria mudança de energia da criatura) destabiliza feio
(65/868/466) — por isso o crédito com cara de recompensa está desligado por
padrão. Clones confirmam a separação de responsabilidades: genomas idênticos
em dois ambientes divergem no `dw` (0,55) permanecendo **bit-iguais no genoma**
(0,000000).

![plasticity](experiments/plasticity/figs/gate.png)

Mais resultados vivem no README de cada experimento: estrutura de cor neutra
9–23× acima de um spawn aleatório (`experiments/tags/`), presentes alinhados
2,6–4,7× acima do chance para parentes (`experiments/proto_culture/`), fogo e
cozinha sobrevivendo enquanto a moradia derruba a população
(`experiments/civilization/`, `experiments/houses/`), países fundadores que
seguram e depois se dissolvem (`experiments/groups/`), cicatrizes de mordida
que cortam a predação mas reduzem a população pela metade
(`experiments/affect/`), e um banco territorial cujo mecanismo move 2296
unidades de energia na seed sobrevivente enquanto seu gate pré-registrado
falha pelo *contexto*, não pelo tratamento (`experiments/territory/`).

## Dificuldades e problemas abertos

### O que foi difícil no caminho

**1. O balanceamento ficou matando os mundos (etapas 8, 9.5, 10, 18).** O
mundo padrão da CLI um dia terminou com 3/86/70 vivos a 5k ticks — um preço
de casa que nunca foi gateado estava comendo a economia em silêncio. A lição,
aprendida duas vezes: os knobs óbvios só mudam a *inclinação* de um
exponencial, então toda correção é um sweep pré-registrado mais uma mudança de
`Config`, nunca um clamp nem um spawn de emergência. "Reporte as métricas,
proponha um parâmetro, deixe o humano decidir" é o que manteve o projeto fora
da sopa de hacks.

**2. A predação se recusou a emergir (etapas 8 → 12.5).** O orçamento de
mordida é `bite_rate × diet`: perto de dieta 0, a carne não paga a mordida, e
a dieta não cresce sem a carne pagar — uma tranca que três experimentos
publicados não abriram. Um ratchet em `bite_rate`/`contact_range` finalmente
tornou a predação *mensurável* (1179 mortes/run vs 43), mas predadores
obrigatórios ainda não aparecem. Ainda aberto (ver abaixo).

**3. Determinismo é uma disciplina constante.** Dois streams de RNG (um
`Generator` do NumPy para draws do lado Python e o stream global do Numba
para kernels); mundos precisam ser avançados **em sequência** porque o stream
do Numba é compartilhado; nenhum draw aleatório dentro de `prange`; save/load
precisa capturar *os dois* streams ou a continuação diverge em silêncio; e
mudança de layout do genoma desloca o stream — por isso caches antigos ficam
velhos e figuras antigas exigem commits antigos. Um bug de determinismo
ultrapassa qualquer feature.

**4. Velocidade num notebook de 2018, sem GPU.** As regras duras existem
porque a alternativa foi medida, não presumida: nenhum loop Python sobre
criaturas, hash espacial em vez de O(N²), buffers pré-alocados dentro do
tick. E como a carga da máquina move ticks/s em 2–4× entre processos, as
comparações são A/B numa única sessão com perfil por fase — um número de outro
dia não significa nada. Minas encontradas no caminho: `@` num `einsum`
atingiu silenciosamente um bug do OpenBLAS (35 ticks/s em vez de milhares), e
dois processos Numba em paralelo se atrapalham ~20×.

**5. O orçamento de 500 parâmetros do cérebro.** Cada sensor novo custa
unidades ocultas: hidden foi 12 → 10 → 9 → 8, e a etapa 24 *encolheu* o genoma
491 → 446 para caber duas entradas novas. Sentidos e capacidade competem de
verdade; algumas etapas pagaram um sentido novo cortando a largura oculta
pela metade.

**6. Arqueologia do formato de save (`FORMAT_VERSION` 1 → 12).** Doze bumps
de layout, cada um rejeitando arquivos velhos com mensagem clara em vez de
remapear colunas em silêncio. A pergunta de design recorrente: o que é
*estado* (salvo, deve continuar bit-idêntico) versus *scratch* (reconstruído
no load — grids, marcas, mapas de flash, cicatrizes vitalícias)?

**7. Publicar falhas.** O hábito mais difícil do projeto: pré-registrar o
gate, rodá-lo como escrito, publicar o FAIL. A moradia falhou duas vezes, o
canal afetivo falhou, o teste de seleção da plasticidade voltou negativo, e o
gate de território falhou no braço de controle. Um gif escolhido a dedo seria
muito mais fácil; os READMEs valem mais justamente por não serem.

**8. Manter o núcleo headless.** O pacote de simulação nunca pode importar
pygame nem matplotlib, e a renderização nunca pode escrever estado — garantido
por testes, porque um único import de conveniência num módulo quente
encerraria a propriedade de "roda em qualquer lugar, inclusive servidores sem
CI".

### Ainda difícil agora

- **A predação é um fio, não uma teia trófica.** ≥99,9% de cada run é
  herbívoro; nenhum predador obrigatório (dieta > 0,5) sobrevive nos
  snapshots. A tranca do desafio 2 não está resolvida, só engrenada.
- **As criaturas ainda giram em círculo.** O `turn_cost` cortou a rotação
  ~10×, mas o movimento reto ainda é raro — o atrator clássico da
  neuroevolução (o pião) só está dentado.
- **Instabilidade do contexto experimental.** Um gate pré-registrado pode
  falhar porque o braço de *controle* extingue (o controle de território morreu
  na seed 7), não porque o tratamento falhe. Consertar isso exige sweeps de
  economia/contexto — o próximo passo honesto, não outro mecanismo.
- **A seleção precifica o aprendizado em vez de promovê-lo.** O η evoluível
  da plasticidade fica abaixo do próprio controle de deriva neutra em 3/3
  seeds; o custo está sendo selecionado contra.
- **A coerência de fundadores decai.** Quatro linhagens fixadas começam com
  exatamente quatro grupos, mas a mutação de cor deriva toda run para ~9
  bandas aos 10k ticks.
- **Lacunas de processo:** sem CI (os testes rodam manualmente antes de cada
  commit), o benchmark é sensível à carga e comparado à mão, a reprodução
  bit-exata de figuras antigas exige commits antigos, e é um projeto de
  mantenedor único.

## Visualizador

`python run.py --seed 42` abre a janela pygame-ce: arraste para mover, roda do
mouse para zoom, clique numa criatura para inspecionar o cérebro (pesos por
camada incluindo a matriz recorrente, atividade de disparo), espaço pausa,
`1-4` muda a velocidade da simulação. A câmera segue a seleção com trilha,
nascimentos e mortes piscam no local, e o painel lateral mostra população,
nascimentos, mortes por causa e energia média ao vivo (sparkline no modo
visão geral). Overlays com `g` (observador de grupos por cor: países,
fronteiras, eventos de contato) e `t` (território: anéis de marca tingidos pela
cor do dono, barras de depósito sobre casas bancadas, linhas laranja de
roubo). O mundo renderiza o terreno sob a camada de comida (lagos azuis,
praias, solo fértil) e a cena toda escurece com o ciclo dia/noite, mostrado
como `light` no HUD.

## O mundo que roda sozinho

`python run.py --daemon --db world.db` inicia um mundo que nunca para:
headless, estado checkpointado num arquivo SQLite comum (stdlib — nenhuma
dependência nova), medidores de timeline a cada 10 ticks, Ctrl-C grava um
checkpoint final. Abra o snapshot mais novo quando quiser com
`python run.py --load-db world.db`, ou retome de onde parou com
`python run.py --load-db world.db --headless --ticks 500`. Uma execução de
120 segundos nesta máquina cobriu 23 570 ticks (população oscilando entre
227–1538, 10 133 nascimentos) num banco de 57 MB com 10 checkpoints mantidos
(~5,2 MB cada); retomar de um checkpoint e continuar N ticks é
**bit-idêntico** a uma execução contínua, exatamente como
`--save`/`--load`.

## Desempenho

Meta: ≥ 30 ticks/s com 2000 criaturas headless (i5-8265U, sem GPU).

| criaturas | ticks/s (defaults da etapa 25, carga 2,5–3,0) |
|---|---|
| 500 | 617 |
| 2000 | 318 |
| 5000 | 162 |

Os números variam 2–4× entre processos conforme a carga da máquina (esta
máquina roda uma sessão desktop); compare A/B na mesma sessão. A passada de
território da etapa 25 (depósito automático, saque/roubo no eat-gate, busca
local de guarda, desconto de conforto no metabolismo, rebuild da marca)
cavalga a estrutura de fases existente — nenhuma fase nova, nenhuma alocação
nova — e o piso segura com ~10× de margem (617/318/162 a carga 2,5–3,0). O
rebuild da etapa 22 (lógica adicional de entrega, podridão, imposto de carga)
deixa o mundo padrão bit-idêntico — `alive_final` 98/693/1166 bate exato com
o HEAD da etapa 21 — e o piso segura com 5× de margem. A etapa 17 adicionou
cinco fases (crescimento de madeira, decaimento de estruturas, coleta,
construção, rebuild de estruturas) que juntas custam ~138 µs/tick ≈ **1,2%**
do tick em N=2000 (perfil: 11/4/33/44/47 µs). A fase `donate` da etapa 16
custa 0,75 µs no default, ~24 µs com presentes ativos; a fase de plasticidade
da etapa 21 ~130 µs/tick ≈ **4%** em N=2000 quando engajada (A/B na mesma
sessão: 301/259 vs 285/253 ticks/s em N=2000, um preço de 2–5% com o piso
intacto). Custos por fase mais antigos estão listados em `AGENTS.md`.
`python -m primordia.bench [--profile]` reproduz a tabela.

## Reproduza tudo

```bash
# experimentos (data/ em cache npz, figuras se regeneram)
python experiments/food_supply/run.py
python experiments/mutation_diversity/run.py
python experiments/predation/run.py
python experiments/lineage/run.py
python experiments/tags/run.py          # etapa 15: cor de linhagem neutra
python experiments/proto_culture/run.py --phase pilot  # etapa 16: doação
python experiments/proto_culture/run.py --phase gate
python experiments/civilization/run.py --phase pilot   # etapa 17: madeira/fogo/casas
python experiments/civilization/run.py --phase gate --house-energy 12
python experiments/civilization/run.py --phase gate --house-energy 1000000  # controle
python experiments/plasticity/run.py --ticks 10000  # etapa 21: ablação Oja (~15 min)
python experiments/plasticity/run.py --phase clones
python experiments/houses/run.py --phase pilot       # etapa 22: casas incrementais (null)
python experiments/groups/run.py --force             # etapa 23: grupos/fundadores (null)
python experiments/affect/run.py                     # etapa 24: estresse+rancor (gate FAIL)
python experiments/territory/run.py --ticks 10000    # etapa 25: banco/roubo/marca (null de contexto)

# stats + gráficos
python run.py --seed 42 --headless --ticks 5000 --stats s42.npz
python plot.py s42.npz --out figs/

# save / load (continuação bit-idêntica)
python run.py --seed 42 --headless --ticks 5000 --save w5k.npz
python run.py --load w5k.npz --headless --ticks 500

# o mundo que roda sozinho (etapa 19: daemon SQLite)
python run.py --daemon --db world.db              # até Ctrl-C
python run.py --load-db world.db                  # observe o snapshot
python run.py --load-db world.db --headless --ticks 500 --save cont.npz

# GIF de demo (o que se vê no topo deste README)
python run.py --seed 42 --headless --ticks 9000 \
    --record docs --record-every 360

# overrides de config JSON (copie overrides.example.json e edite os campos;
# nome de campo desconhecido é rejeitado em voz alta)
python run.py --headless --ticks 1000 --config overrides.example.json
```

Cada diretório de experimento tem as mesmas quatro coisas: um `run.py`
executável, dados brutos em cache num `data/` ignorado pelo git, figuras
geradas em `figs/`, e um `README.md` com a pergunta, o método, os números e
as limitações.

## Estrutura do projeto

```
primordia/
  config.py world.py genome.py brain.py sensors.py
  step.py     stats.py   io.py       bench.py ecology.py
  cycles.py   # etapa-10: leis de luz/dia-noite/estações
  database.py # etapa-19: armazenamento SQLite (checkpoints + timeline)
  observer.py # etapa-23: observador de grupos por cor (off-tick)
  render/     # visualizador pygame + gravador de GIF (nunca importado pelo núcleo)
tests/        # 311 testes: determinismo, invariantes, save/load, guards
experiments/  # um script + README por experimento (índice: experiments/README.md)
run.py        # CLI: --seed --headless --ticks --load --save --stats
              #      --config --record --screenshot --select --daemon --db
```

## Roadmap (25 etapas)

Todas as etapas planejadas estão implementadas. A etapa 20 (esboço antigo de
território) foi absorvida pela etapa 25.

| Etapa | O que chegou | Veredito |
|---|---|---|
| 1 | Mundo + grid de comida, tick headless, bench | 11k ticks/s nos tamanhos de spawn |
| 2 | MLP do cérebro + sensores de raios (Numba) | perfil tirado: perceive/think dominam |
| 3 | Genoma, mutação, reprodução por energia | slots em free-list, shape fixado |
| 4 | Visualizador Pygame: câmera, zoom, painel do cérebro | 26–47 fps em N=2000 |
| 5 | Predação como traço de dieta evoluível | funciona mecanicamente, inerte (resultado 3) |
| 6 | Gravador de stats + gráficos matplotlib | ~0,5 ms/tick, fora do caminho quente |
| 7 | Save/load (npz v1) + log de genealogia | continuação provada bit-idêntica |
| 8 | Experimentos, GIF de demo, README, publicação no GitHub | 4 experimentos, gates definidos |
| 9 | "Mundo que ensina": patches, sentidos ricos, Elman | GATE aceito pelo usuário |
| 9.5 | Balanço: economia de comida, `turn_cost`, zoom | seleção existe (a fome morde) |
| 10 | Ambiente: terreno, água, dia/noite, estações | nichos medidos; crescimento recalibrado |
| 11 | Biologia: sexo, senescência, carniça | GATE passa (3 seeds × 10k) |
| 12 | Ecologia: classes de dieta, clusters, barreira | especiação-lite; carne dobra a barreira |
| 13 | Canal de sinais: gate de emissão + cheiros | seleção fecha gates caros; default off |
| 14 | Difusão de feromônio (Jacobi, conserva massa) | trilhas são físicas, ainda não lidas |
| 15 | Cor de linhagem neutra + sensor de similaridade | estrutura 9–23× acima do nulo |
| 16 | Saída de doação (vizinho mais próximo) | GATE passa; presentes alinhados a parentes 2,6–4,7× |
| 17 | Civilização: madeira, fogueiras, casas | fogo sobrevive; casas derrubam a pop |
| 18 | Sweep de balanço dos preços padrão | mundo padrão termina 1163/1170/821 |
| 19 | Daemon + checkpoints/timeline SQLite | 23,5k ticks em 120 s, resume bit-idêntico |
| 20 | *(esboço antigo de território — absorvido pela 25)* | — |
| 21 | Plasticidade de Oja num `dw` separado, η evoluível | controle nunca vence; η selecionado contra; default off |
| 22 | Casas erguidas em entregas, podridão, imposto de carga | null honesto #2; default `houses_enabled = false` |
| 23 | Observador off-tick de grupos por cor + cenário fundador | 4 grupos seguram, depois derivam para ~9 bandas |
| 24 | Entradas afetivas: estresse + rancor vitalícios | corta predação, reduz pop pela metade; HARD GATE FAIL, default off |
| 25 | Território: depósito automático, roubos guardados, marca + conforto | mecanismo funciona (2296 bancados); gate falha pelo contexto |

Problemas abertos conhecidos — predação ainda como um fio, criaturas ainda
girando em círculo, contextos de experimento ainda frágeis — estão documentados
acima em vez de escondidos.

## Licença

MIT — ver [LICENSE](LICENSE).
