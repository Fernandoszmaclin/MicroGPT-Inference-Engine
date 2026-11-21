# Executar o MicroGPT

Use Python 3.11 ou mais recente. Na raiz do repositório, crie um ambiente local e instale o NumPy.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use `.\.venv\Scripts\python.exe` no lugar de `python` nos comandos abaixo se o ambiente virtual estiver desativado.

**Linux e macOS**

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Gerar texto

```sh
python -m microgpt make-demo demo.npz
python -m microgpt generate demo.npz --prompt MicroGPT --max-new-tokens 24 --backend numpy
python -m microgpt generate demo.npz --prompt MiniApps --max-new-tokens 24 --backend numpy
```

Cada comando `generate` imprime o prompt seguido da continuação, como nas duas linhas abaixo.

```text
MicroGPT: ola, mundo!
MiniApps: viva o codigo!
```

A temperatura padrão `0` escolhe o token de maior pontuação. Use `--temperature 0.7 --seed 42` para experimentar amostragem reproduzível no mesmo ambiente. A geração termina ao escolher o token de fim de sequência, esgotar o limite de novos tokens ou preencher a janela de contexto. A janela da demo comporta 26 tokens, incluindo o token de início inserido pelo CLI.

Para usar a interface fora da raiz do repositório, instale o pacote com `python -m pip install .` a partir da raiz. Com o ambiente virtual ativado, você pode chamar `microgpt` no lugar de `python -m microgpt`.

## Pesos e alcance da demonstração

O comando `make-demo` produz o checkpoint localmente, sem baixar pesos. O código inicializa as matrizes de embeddings, atenção e MLP de forma determinística com seed 2027. A projeção para o vocabulário é ajustada por regressão ridge a partir dos estados do decoder nas duas frases de exemplo. O checkpoint usa o mesmo formato `.npz` que o motor lê para inferência, com configuração, vocabulário e tensores `float32`.

Os resultados documentados correspondem às duas frases usadas no ajuste. O vocabulário é composto de caracteres que ocorrem nessas frases, além dos tokens de início e fim. Caracteres fora do vocabulário geram erro. Outros prompts válidos podem produzir textos sem sentido. O motor usa o formato de checkpoint do próprio projeto e recalcula o prefixo inteiro a cada passo de geração.

## Atenção em C

O backend C é opcional e executa o produto escalar escalado, a máscara causal, o softmax e a combinação dos valores da atenção. O restante da inferência roda em Python e NumPy.

No Linux ou macOS, use um compilador C disponível no terminal. No Windows, execute em um terminal **x64 Native Tools Command Prompt for Visual Studio** com as ferramentas C++ e o Windows SDK instalados. Use o Python do ambiente virtual nesse terminal.

```sh
python -m microgpt.build_native
python -m microgpt generate demo.npz --prompt MicroGPT --max-new-tokens 24 --backend c
```

Use `--backend auto` para selecionar C quando a biblioteca puder ser carregada e recorrer ao NumPy caso contrário. `--backend numpy` sempre usa a implementação de referência. `--backend c` exige a biblioteca nativa e informa um erro se ela estiver indisponível.

## Validar

```sh
python -m unittest discover -s tests -v
```

A suíte cobre a comparação dos logits com uma implementação independente em `float64`, a invariância do prefixo causal, a tokenização, os checkpoints e a geração pelo CLI. Sem a biblioteca C, a suíte pula os três testes que dependem dela. Depois da compilação, rode a mesma suíte para verificar a equivalência numérica e a geração com os dois backends.
