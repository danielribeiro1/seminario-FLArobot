# INF0429 - Robotics

## Seminário - FLArobot, Interação Homem-Robô com drone Tello

### Integrantes
- 202105833 - Arthur Jung Barreto;
- 202105835 - Daniel Ribeiro da Silva;
- 202105844 - Gabriel van der Schmidt;
- 202105848 - Hugo Fernandes Silva.

## Relatório Final

Relatório descrevendo toda a implementação: [Relatório Final - FLArobot](Relatório.pdf).

## Apresentações

Apresentações realizadas durante e após a implementação do projeto: [Apresentações](https://drive.google.com/drive/folders/12nAjPHUr_3Wk-I5l_NAJ49CnR4sdqFbL?usp=sharing).

## Vídeos

Vídeos mostrando a implementação na prática: [Vídeos](https://drive.google.com/drive/folders/1IUTySIOkhd4fTqNvGjHl3K1Qny4y-R4l?usp=sharing).

## Códigos

Scripts do repositório (em [code](code)):

1. [anottate_data](code/annotate_data.py): Código para realizar captura e anotação de imagens;
2. [train_conv1d](code/train_conv1d.py): Código para treinar o modelo CNN;
3. [train_mlp](code/script_mlp.py): Código para treinar o modelo MLP;
4. [script_conv1d](code/script_conv1d.py): Código para realizar inferência com o modelo CNN treinado;
5. [script_mlp](code/script_mlp.py): Código para realizar inferência com o modelo MLP treinado.

## Dados

Dados usados para treinar os modelos: [Dados](https://drive.google.com/file/d/1BKxSd8WXPLu_BiryX7B-QhOR9M9l9pVv/view?usp=sharing).

## Modelos

Modelos do repositório (em [models](models)):

1. [conv1d](models/conv1d.pth): Modelo CNN treinado;
2. [mlp_classifier](models/mlp_classifier.joblib): Modelo MLP treinado.

## CNN architecture

```mermaid
graph TD
    A[Input: 1D Sequence] --> B[Conv1D Layer 1: in_channels=1, out_channels=16, kernel_size=3, padding=1]
    B --> C[ReLU Activation]
    C --> D[MaxPool1D Layer 1: kernel_size=2, stride=2]
    D --> E[Conv1D Layer 2: in_channels=16, out_channels=32, kernel_size=3, padding=1]
    E --> F[ReLU Activation]
    F --> G[MaxPool1D Layer 2: kernel_size=2, stride=2]
    G --> H[Flatten Layer]
    H --> I[Fully Connected Layer 1: in_features=32 * input_size//2//2, out_features=64]
    I --> J[ReLU Activation]
    J --> K[Fully Connected Layer 2: in_features=64, out_features=9]
    K --> L[Output: Classes]
