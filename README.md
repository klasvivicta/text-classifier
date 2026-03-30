# Text Classifier

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 1.2.0`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Projektöversikt

Det här projektet är en textklassificeringslösning byggd i Kedro. Syftet är att kunna klassificera korta texter i fördefinierade intressekategorier, men också att utforska mer semantiska metoder och att upptäcka kandidater till nya kategorier.

Lösningen är uppdelad i flera pipeline-spår så att olika metoder kan jämföras utan att blandas ihop:

- `data_load` för att läsa in och strukturera textdata.
- `model_training` för en klassisk baseline med TF-IDF och Logistic Regression.
- `embedding_training` för en mer semantisk modell där meningar först omvandlas till embeddings och därefter klassificeras med en klassisk maskininlärningsmodell.
- `category_discovery` för att undersöka om det finns grupper av meningar som kan tyda på att en ny kategori saknas.

## Metod

Projektet använder två huvudsakliga angreppssätt för klassificering.

Det första är en traditionell bag-of-words-liknande metod där texten representeras med TF-IDF. Där ges högre vikt åt ord som är informativa i relation till resten av materialet. Dessa vektorer används sedan som indata till en Logistic Regression-modell. Det här fungerar som en tydlig och lättolkad baseline, och gör det möjligt att förstå vilka ord och fraser som driver klassificeringen.

Det andra angreppssättet bygger på embeddings från en liten flerspråkig språkmodell. I stället för att representera text som en lista av ordvikter representeras varje mening som en tät numerisk vektor som försöker fånga betydelse och semantisk likhet. Ovanpå dessa embeddings tränas en klassisk classifier, även här Logistic Regression. Det ger ofta bättre generalisering än TF-IDF när två meningar betyder liknande saker men använder olika ord.

Normalisering av embeddings används för att göra vektorerna mer jämförbara. Det innebär att fokus hamnar mer på riktning i det semantiska rummet än på absolut storlek, vilket ofta passar bra för både likhetsberäkning och klassificering.

## Baseline: TF-IDF + Logistic Regression

Baseline-modellen är avsiktligt enkel. Den används som referensmodell för att snabbt kunna svara på frågor som:

- Hur bra fungerar en lättviktig klassisk textmodell på datat?
- Vilka ord verkar vara mest informativa?
- Hur mycket vinner en embedding-baserad metod jämfört med en enklare representation?

För att minska brus används stoppord. Det gör att mycket vanliga småord inte får onödigt stort genomslag i representationen, samtidigt som mer meningsbärande ord och fraser får större betydelse.

## Embedding-baserad klassificering

I embedding-spåret används en sentence-transformer för att skapa en vektor per mening. Den vektorn beskriver meningens innehåll i ett semantiskt rum där liknande meningar hamnar nära varandra. Därefter tränas en Logistic Regression-modell på dessa vektorer.

Den här metoden är särskilt intressant när man vill:

- fånga semantisk likhet snarare än exakt ordöverlapp
- hantera korta meningar bättre
- skapa en grund för vidare arbete med mer moderna språkmodeller

Det här är också en bra mellanväg mellan klassisk ML och generativa språkmodeller. Man får en modern textrepresentation utan att behöva finjustera en full LLM för själva klassificeringen.

## Upptäckt av nya kategorier

Projektet innehåller även ett explorativt spår för kategoriupptäckt. Tanken är inte att automatiskt skapa sanna nya etiketter, utan att identifiera grupper av meningar som verkar tillhöra något gemensamt tema men som inte passar rent in i de befintliga kategorierna.

Detta görs i två steg:

- meningarna omvandlas till embeddings
- embeddings klustras med DBSCAN

DBSCAN används eftersom algoritmen inte kräver att antalet kluster bestäms i förväg. Den letar i stället efter täta grupper i embedding-rummet och kan samtidigt märka ut avvikande punkter som brus. Det gör metoden lämplig när man vill hitta möjliga nya teman i textdata.

Efter klustringen analyseras varje kluster i relation till de befintliga etiketterna. Om ett kluster domineras tydligt av en redan känd kategori tolkas det som ett befintligt tema. Om ett kluster däremot är mer blandat, eller inte tydligt passar in under en befintlig label, markeras det som en kandidat till ny kategori. Projektet tar också fram representativa texter och frekventa nyckelord för att göra sådana kandidater lättare att tolka manuellt.

Det betyder i praktiken att lösningen inte “uppfinner” nya kategorier helt automatiskt. Den föreslår snarare semantiskt sammanhållna grupper som en människa sedan kan granska och eventuellt omvandla till en ny taxonomisk kategori, till exempel ett nytt tema som socialt umgänge, familjeliv eller andra tidigare odefinierade intressen.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the files `tests/test_run.py` and `tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
