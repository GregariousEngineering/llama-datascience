# Llama Data Science 🦙

**Your Personal AI Data Scientist on the Command Line!**

Ever wished you had a senior data scientist on speed dial? Now you do! Llama Data Science is a powerful command-line tool that connects you with a brilliant AI data scientist, ready to analyze your data, answer your toughest questions, and even whip up visualizations in seconds.

Harness the power of Large Language Models running locally on your machine to turn data into insights, instantly.

## Why You'll Love It 🚀

-   **Interactive AI Chat**: Go beyond simple queries. Have a full conversation with your AI partner to explore your data from every angle.
-   **Safe & Secure**: All Python code is executed in a secure, isolated sandbox. Your data and your machine are always safe.
-   **Instant Visualizations**: Ask for a plot, and get it! The AI uses libraries like Matplotlib to generate charts and save them directly to your disk.
-   **Blazing Fast Answers**: Get immediate responses to one-off questions without leaving your terminal.
-   **BYOM (Bring Your Own Model)**: Works with any Ollama-compatible model. You have complete control over the AI brainpower you want to use.

## Getting Started

### 1. Prerequisites

-   **Python 3.x**
-   **Ollama**: Make sure you have [Ollama](https://ollama.com) installed and running. You'll also need at least one model pulled.

    ```bash
    # Example: Pull the Mistral model
    ollama pull gpt-oss
    ```

### 2. Installation

Install the necessary Python packages using pip:

```bash
pip install llm_sandbox ollama termcolor
```

## How to Use

Unleash your AI data scientist in two powerful modes.

### Quick Question Mode

Have a single, burning question? Get a fast answer.

```bash
./llama_datascience.py "What are the column names and data types in my dataset?" --data-file /path/to/your/data.csv
```

### Interactive Chat Mode 💬

For deeper analysis, fire up the interactive chat. This is where the magic happens!

```bash
./llama_datascience.py --data-file /path/to/your/data.csv
```

You'll be greeted by the AI and can start asking questions. To end the session, just type `exit` or `quit`.

## Power-User Options

Customize your experience with these command-line arguments:

| Argument              | Description                                                               | Default                     |
| --------------------- | ------------------------------------------------------------------------- | --------------------------- |
| `question`            | A single question to ask. Omit for interactive mode.                      | `None`                      |
| `--data-file`         | Path to your CSV data file.                                               | `None`                      |
| `--expert`            | The LLM model and temperature. Format: `model_name:temp`.                 | `gpt-oss:0.5`               |
| `--max-steps`         | Max number of tool calls (code executions) the AI can make per turn.      | `20`                        |
| `--execution-timeout` | Max time (in seconds) for a single code execution step.                   | `600`                       |
| `--verbose`           | See the AI's thought process and reasoning.                               | `False`                     |
| `--thinking`          | Enable the model's "thinking" output if supported.                        | `False`                     |
| `--write-convo`       | Save the entire conversation to a `.convo` file.                          | `False`                     |

---

```bash
gregarious@hal9000:~/code/llama-datascience$ ./llama_datascience.py --data-file traffic_volume_000256_2004_2024.csv "What is the best time to travel South through the intersection?"
Expert: gpt-oss (temp: 0.5)

--- Expert Reasoning Step 1/20 ---
> Expert wants to use tool: 'python_datascience'
> Reason for tool selection: Load and inspect data
> Code:
import pandas as pd
import numpy as np

# Load data
try:
    df = pd.read_csv('/sandbox/data.csv')
except Exception as e:
    print('Error loading CSV:', e)
    df = None

print('Columns:', df.columns.tolist() if df is not None else None)
print('Head:')
print(df.head() if df is not None else None)

> Code execution complete.
output:
Columns: ['Count Date', 'Dir', '0h', '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h']
Head:
   Count Date Dir  0h  1h  2h  3h   4h  ...   17h   18h  19h  20h  21h  22h  23h
0  12/01/2024   P  91  45  32  23   21  ...   837   575  445  358  198  149   63
1  12/01/2024   S  49  30  19  28   42  ...   783   535  359  199  164   89   48
2  12/02/2024   P  54  27  20  26   49  ...  1758   794  486  326  237  191  111
3  12/02/2024   S  18  17  15  36  154  ...   924   506  265  214  163  104   54
4  12/03/2024   P  54  27  17  33   52  ...  1745  1017  584  456  331  145  105

[5 rows x 26 columns]


--- Expert Reasoning Step 2/20 ---
> Expert wants to use tool: 'python_datascience'
> Reason for tool selection: Compute average hourly counts for South direction and find best hour
> Code:
import pandas as pd
import numpy as np

# Load data
try:
    df = pd.read_csv('/sandbox/data.csv')
except Exception as e:
    print('Error loading CSV:', e)
    df = None

# Ensure numeric columns
hour_cols = [col for col in df.columns if col.endswith('h')]
# Convert to numeric
for col in hour_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter South direction
south_df = df[df['Dir'] == 'S']

# Compute average per hour across all South rows
avg_counts = south_df[hour_cols].mean()

# Find hour with minimum average
best_hour = avg_counts.idxmin()
print('Average counts per hour for South direction:')
print(avg_counts)
print('\nBest hour (lowest average traffic):', best_hour)

# Also show min count across all South rows for that hour
min_count = south_df[best_hour].min()
print('Minimum count on best hour across all days:', min_count)

> Code execution complete.
output:
Average counts per hour for South direction:
0h       40.570433
1h       25.991288
2h       22.957110
3h       34.281598
4h       97.354108
5h      349.480231
6h      762.930438
7h     1107.911272
8h      900.093553
9h      686.656346
10h     630.000670
11h     644.783273
12h     646.281866
13h     645.476880
14h     654.760488
15h     681.876960
16h     741.068489
17h     734.114194
18h     533.185096
19h     328.717732
20h     260.428897
21h     208.710763
22h     134.479426
23h      70.539204
dtype: float64

Best hour (lowest average traffic): 2h
Minimum count on best hour across all days: 4


--- Expert Reasoning Step 3/20 ---
**Best time to travel South**

The hour with the lowest average vehicle count for the South‑bound direction is **2 h** (the 2 :00 – 3 :00 AM slot).  
Across all days in the data set, the average count for that hour is about **23 vehicles**, and the smallest single‑day count recorded in that hour is **4 vehicles**.

So, if you want the least traffic when heading South through the intersection, aim for the 2 :00 – 3 :00 AM window.
```

---

Happy Analyzing!
