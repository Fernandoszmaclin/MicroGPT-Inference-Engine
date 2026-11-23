# MicroGPT Inference Engine

O MicroGPT Inference Engine implementa, em Python e NumPy, a geração de texto de um pequeno modelo da família GPT. O motor carrega o vocabulário, a configuração e os pesos de um arquivo `.npz`. A partir de um texto inicial, o decodificador combina representações dos caracteres e de suas posições para prever o próximo caractere. A atenção causal permite que cada posição use apenas o contexto disponível até ela.

O checkpoint de demonstração é gerado localmente com a projeção de saída ajustada a duas frases. O modelo reproduz essas frases a partir dos respectivos prefixos, gerando `MicroGPT: ola, mundo!` e `MiniApps: viva o codigo!`. O vocabulário de texto se limita aos caracteres das duas frases. Mesmo dentro desse vocabulário, outros textos iniciais podem gerar texto incoerente.

O [guia de uso](https://github.com/Fernandoszmaclin/MicroGPT-Inference-Engine/blob/main/docs/USO.md) descreve como gerar o checkpoint, executar a demonstração e compilar a atenção em C na versão completa do motor.
