# Dataset IBM Telco Customer Churn

O CSV não é incluído na versão atual por cautela enquanto a autorização específica de redistribuição dos dados não estiver esclarecida. A versão anterior já o incluiu no histórico público; sua remoção da árvore atual não apaga esse histórico. O dataset é atribuído à IBM. O repositório da IBM tem licença Apache 2.0, mas não se pressupõe aqui que ela resolva as condições específicas aplicáveis ao CSV.

## Obtenção e reprodução

A fonte consultada é o arquivo de amostra [Telco-Customer-Churn.csv no repositório IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). Consulte os termos da fonte antes de usá-lo. O script abaixo baixa o arquivo para uso local, verifica o hash da fonte e converte para a representação textual canônica usada no projeto: em `MonthlyCharges`, valores inteiros recebem o sufixo `.0`. Os dados e a ordem dos registros não são alterados. O arquivo resultante é ignorado pelo Git.

Execute na raiz do projeto, com Python 3.12:

```python
import csv
import hashlib
import io
import urllib.request
from pathlib import Path

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
with urllib.request.urlopen(url, timeout=30) as response:
    raw = response.read()
if hashlib.sha256(raw).hexdigest() != "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91":
    raise ValueError("Arquivo da fonte mudou; revise sua procedência antes de continuar.")
rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
index = rows[0].index("MonthlyCharges")
for row in rows[1:]:
    if "." not in row[index]:
        row[index] += ".0"
out = io.StringIO(newline="")
csv.writer(out, lineterminator="\n").writerows(rows)
data = out.getvalue().encode("utf-8")
if hashlib.sha256(data).hexdigest() != "a0ea74c48c1063be89a73ce3fb868593604c0129f2bf581934f04983d728b3b5":
    raise ValueError("A representação canônica não corresponde ao hash esperado.")
Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv").write_bytes(data)
```

Salve o trecho temporariamente como, por exemplo, `obter_dados.py` na raiz e execute `python obter_dados.py`. Em seguida execute a auditoria, treinamento e testes indicados no README principal. O arquivo contém 7.043 registros e 21 colunas.
