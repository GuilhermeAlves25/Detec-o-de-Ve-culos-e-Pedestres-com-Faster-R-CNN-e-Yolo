# Detecção de Objetos para Condução Autônoma com Faster R-CNN e YOLO

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Este repositório contém a implementação e análise de modelos de detecção de objetos para o dataset KITTI, com foco em aplicações para veículos autônomos. O projeto explora e compara a arquitetura de dois estágios **Faster R-CNN** (com diferentes backbones) e a arquitetura de estágio único **YOLOv8**.

## Índice

1.  [Objetivos do Projeto](#objetivos-do-projeto)
2.  [O Dataset KITTI e seu Desafio](#o-dataset-kitti-e-seu-desafio)
3.  [Metodologia e Pipeline de Otimização](#metodologia-e-pipeline-de-otimização)
4.  [Como Executar o Código](#como-executar-o-código)
5.  [Resultados e Análises](#resultados-e-análises)
6.  [Conclusão](#conclusão)
7.  [Estrutura dos Arquivos](#estrutura-dos-arquivos)

## Objetivos do Projeto

Este projeto foi desenvolvido para cumprir os seguintes objetivos de aprendizado:
-   Entender a arquitetura de duas etapas do Faster R-CNN (RPN + classificação).
-   Implementar a Region Proposal Network (RPN) e visualizar suas propostas.
-   Explorar diferentes backbones para extração de características (ResNet-50 vs. MobileNetV3).
-   Analisar o desempenho dos modelos em diferentes condições de iluminação и oclusão.
-   Comparar métricas de avaliação como AP por classe, recall, precisão e tempo de inferência.

## O Dataset KITTI e seu Desafio

Utilizamos o dataset KITTI, um dos principais benchmarks para visão computacional em condução autônoma. Ele é composto por **7.478 imagens** de cenas de trânsito reais.

A principal característica e desafio deste dataset é o **severo desbalanceamento de classes**. Uma análise do nosso conjunto de treino revelou a seguinte distribuição:

Carro: 23.007 objetos
Pedestre: 3.599 objetos
Ciclista: 1.295 objetos

Este desbalanceamento, com a classe `Car` sendo massivamente dominante, impacta diretamente o treinamento e o desempenho dos modelos nas classes minoritárias.

## Metodologia e Pipeline de Otimização

Para garantir um treinamento eficiente e viável no ambiente Google Colab/Kaggle, foi implementado um pipeline de otimização em várias etapas:

1.  **Limpeza de Dados:** Foi executado um script para remover arquivos duplicados ou órfãos, garantindo a consistência entre imagens e anotações.
2.  **Cópia Local:** O dataset foi copiado do Google Drive/Kaggle Datasets para o disco local do ambiente de execução, eliminando gargalos de leitura de dados.
3.  **Pré-processamento:** As imagens de alta resolução foram redimensionadas para `600x600 pixels` e as anotações foram ajustadas correspondentemente. Este passo foi feito *antes* do treinamento para reduzir a carga computacional a cada época.
4.  **Carregamento Paralelo:** O `DataLoader` do PyTorch foi configurado com `num_workers=2` para preparar os lotes de dados em paralelo, mantendo a GPU sempre ocupada.

## Como Executar o Código

O projeto está contido em um notebook Jupyter (`.ipynb`).

### Pré-requisitos
- Python 3.9+
- GPU NVIDIA com CUDA (para treinamento local) ou um ambiente como Google Colab/Kaggle.

### Instalação
As principais bibliotecas utilizadas podem ser instaladas com o seguinte comando:
```bash
pip install torch torchvision ultralytics opencv-python matplotlib tqdm torchmetrics
```

### Execução
1.  Abra o notebook em um ambiente compatível (Google Colab, Kaggle, ou localmente com Jupyter).
2.  Certifique-se de que o dataset KITTI (versão limpa) está acessível no caminho especificado na célula de preparação de dados.
3.  Execute as células em ordem.

## Resultados e Análises

### 1. Comparação de Backbones: ResNet-50 vs. MobileNetV3 (5 Épocas)
Para avaliar o impacto do backbone, ambos os modelos foram treinados por 5 épocas.

| Métrica | Faster R-CNN com **ResNet-50** | Faster R-CNN com **MobileNetV3** |
| :--- | :--- | :--- |
| **Tempo médio por Época**| ~18-20 minutos | **~5-8 minutos** |
| **mAP (0.50:0.95)**| **0.5284** | 0.3763 |
| **Recall Médio**| **0.6202** | 0.4601 |

**Análise:** O ResNet-50 demonstrou ser um extrator de características mais poderoso, alcançando maior precisão e recall no mesmo número de épocas, ao custo de um tempo de treinamento 3x maior.

### 2. Comparação Final de Modelos: Faster R-CNN vs. YOLOv8 (20 Épocas)

| Métrica | **Faster R-CNN (ResNet-50)** | **YOLOv8n** | Vantagem |
| :--- | :--- | :--- | :--- |
| **Tempo por Época** | ~18-20 minutos | **~45 segundos** | **YOLOv8** |
| **mAP (0.50:0.95)** | **0.5438** | 0.525 | **Faster R-CNN** |
| **Recall Médio** | 0.6227 | **0.705** |  **YOLOv8** |
| **AP para `Pedestrian`**| N/A | **0.393** | **YOLOv8** |
| **AP para `Cyclist`** | N/A | **0.468** | **YOLOv8** |

### 3. Análise Visual de Casos Específicos

**Análise:** Em testes qualitativos, o YOLOv8, com seu recall superior, demonstrou maior capacidade de detectar objetos parcialmente ocluídos, enquanto o Faster R-CNN produziu caixas delimitadoras ligeiramente mais precisas para os objetos que detectou.

## Conclusão

O projeto demonstrou com sucesso a implementação e o treinamento de modelos de detecção de objetos de última geração. A análise comparativa revelou um claro trade-off entre as arquiteturas:

-   O **Faster R-CNN com ResNet-50** alcançou a **maior precisão (mAP)**, mas se mostrou computacionalmente caro e sensível ao desbalanceamento de classes, não conseguindo aprender a detectar classes minoritárias em 20 épocas.
-   O **YOLOv8** se mostrou **extremamente rápido, com maior recall e mais robusto** ao desbalanceamento, conseguindo detectar todas as classes.

Para a aplicação em condução autônoma, onde a **velocidade em tempo real** e o **alto recall** (não perder nenhum objeto) são cruciais para a segurança, o **YOLOv8 se apresenta como a escolha mais pragmática e balanceada**.

