# AgroShield AI

Sistema inteligente de monitoramento agrícola utilizando Inteligência Artificial e benchmark computacional entre os padrões numéricos int8 e float64 em sistemas operacionais embarcados.

---

## ⚠️ Aviso Importante — Dados Simulados

> **Os dados utilizados neste projeto são simulados e sintéticos.** O dataset foi gerado artificialmente com o objetivo exclusivo de viabilizar a comparação de desempenho computacional entre os padrões numéricos `float64` e `int8` em uma rede neural MLP. Os valores **não representam medições reais** de condições agrícolas e **não devem ser utilizados para tomadas de decisão** no mundo real. O projeto tem finalidade exclusivamente acadêmica.

---

## Objetivo

O projeto tem como objetivo analisar o impacto de diferentes padrões numéricos no desempenho computacional de redes neurais MLP aplicadas ao monitoramento agrícola inteligente.

A aplicação simula um sistema de previsão agrícola capaz de:

- monitorar condições do solo;
- prever riscos agrícolas;
- gerar inferências utilizando Inteligência Artificial;
- comparar desempenho computacional entre `float64` e `int8`.

---

## Tecnologias Utilizadas

- Python
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Estrutura do Projeto

```bash
global-solution-os/
│
├── codigo/
├── dataset/
├── graficos/
├── prints/
├── relatorio/
└── README.md
```

---

## Funcionalidades

- Leitura de dataset agrícola
- Treinamento de rede neural MLP
- Geração de gráficos de accuracy e loss
- Benchmark computacional
- Comparação entre `float64` e `int8`
- Medição de tempo de processamento

---

## Resultados Esperados

O projeto demonstra:

- diferenças de desempenho computacional;
- impacto do tipo numérico na precisão;
- tempo de processamento;
- aplicações de IA em sistemas embarcados.

---

## Como Executar

1. Instale as bibliotecas:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

2. Execute o projeto:

```bash
cd codigo
python main.py
```

Os gráficos gerados serão salvos automaticamente na pasta `graficos/`.

---

## Autores

- Gabriel Borba — RM553187
- Gabriel Souza Fiore — RM553710
- Gustavo Gouvêa Soares — RM553842
- Pedro Henrique Mello Silva Alves — RM554223
- Guilherme Santiago — RM552321
