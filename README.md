# :speech_balloon: Analyzing Chat Messages for Future Automation

## Introduction
### About this Project

This project is based on a dataset derived from a gaming community that operates within a simulated medical emergency service in a science fiction universe. The gameplay involves scenarios where a player character, referred to as the "client", experiences incapacitation or isolation. In response, a coordinated team of players, termed the "responding team", is mobilized to provide assistance. Communication between the responding team and the client is facilitated through a text-based chat system established for each incident.

The objective of this analysis is to identify recurring patterns and commonalities in the chat communications. This exploration assists the community's leadership in determining the feasibility of automating certain repetitive messages, thereby streamlining operations and enhancing response efficiency.

Despite the dataset originating from the context of a video game, the analytical techniques and insights gained from this project are universally applicable to a wide range of industries. The core challenge addressed in this analysis - identifying and automating repetitive communication patterns - is prevalent in numerous settings where service representatives interact with clients.

The methodology employed here can be adapted to enhance customer service efficiency, reduce response times, and improve client satisfaction in various sectors, including healthcare, customer support, and tech support.

### About the Dataset

The dataset utilized in this analysis is strictly for internal use due to containing in-game usernames. As a commitment to respecting privacy and upholding ethical standards in data handling, the dataset itself will not be shared publicly.

In any outputs or examples provided within this project documentation or related presentations, identifiable information has been anonymized or replaced with placeholders such as `[Organization]`, `[PlayerName]`, `[Location]`, and similar terms. The findings and methodologies are shared openly for educational and demonstrative purposes while ensuring the privacy and anonymity of the individuals involved.

#### The Dataset's Description

- The dataset has 6 columns and 16&nbsp;509 rows.
- Data coverage:
  - Start date: June 18, 2023
  - End date: March 21, 2024

 The dataset's columns listed below are the result of normalizing data from `json` format and renaming the columns for clarity. Detailed description for each column:

| Column name            | Description                                                 | Type     |
|------------------------|-------------------------------------------------------------|----------|
| ID                     | Unique identifier for each message within the database      | Text     |
| contents               | Text content of the chat message sent between players       | Text     |
| created                | Date and time when the message was recorded in the system   | Datetime |
| emergency ID           | Identifier linking the message to a specific emergency case | Text     |
| sender ID              | Unique identifier of the player who sent the message        | Text     |
| message sent timestamp | Date and time when the message was sent by the player       | Datetime |

### The Tools I Use:

- **Python:** Programming language for data manipulation and analysis.
- **Pandas:** Python library for efficient data cleaning and transformation.
- **NLTK (Natural Language Toolkit):** Python toolkit for natural language processing and text analysis.
- **Scikit-learn:** Machine learning library for building predictive models.
- **Matplotlib:** Visualization tool for creating plots and graphs.
- **Jupyter Notebook:** Interactive environment for documenting data analysis.

### Project Outcome

This project aimed to analyze and optimize the communication process within a text chat system by identifying common themes and frequently used messages for further automation. 

Through clustering and topic modeling, I discovered that the majority of manual messages are related to sending friend requests and party invites. By automating these and other frequently used messages, we can streamline up to **20%** of the current manual communication, significantly reducing the team's workload and improving efficiency. The insights gained from this analysis provide a clear path forward for enhancing the community's messaging system through targeted automation.

## Table of Contents

- [1. Setting up the Environment](#1-setting-up-the-environment)
- [2. Loading and Preprocessing the Data](#2-loading-and-preprocessing-the-data)
   - [2.1. Importing Libraries](#21-importing-libraries)
   - [2.2. Loading the Dataset](#22-loading-the-dataset)
   - [2.3. Renaming Columns](#23-renaming-columns)
   - [2.4. Changing Column Types](#24-changing-column-types)
   - [2.5. Removing System-generated Messages](#25-removing-system-generated-messages)
   - [2.6. Exploratory Attempt to Filter Non-English Messages](#26-exploratory-attempt-to-filter-non-english-messages)
   - [2.7. Text Preprocessing](#27-text-preprocessing)
- [3. Vectorization](#3-vectorization)
- [4. Initial Analysis](#4-initial-analysis)
   - [4.1. Clustering](#41-clustering)
   - [4.2. Topic Modeling with NMF](#42-topic-modeling-with-nmf)
   - [4.3. Evaluation of the First Results](#43-evaluation-of-the-first-results)
- [5. Fine-Tuning Clustering](#5-fine-tuning-clustering)
   - [5.1. Removing Identified Pre-written Messages](#51-removing-identified-pre-written-messages)
   - [5.2. Determining the Optimal Number of Clusters](#52-determining-the-optimal-number-of-clusters)
- [6. Final Clustering Execution](#6-final-clustering-execution)
   - [6.1. Running the Model](#61-running-the-model)
   - [6.2. Cluster Analysis](#62-cluster-analysis)
- [7. Project Conclusion and Recommendations](#7-project-conclusion-and-recommendations)


## Project Workflow
### 1. Setting up the Environment

The following command installs the Python libraries required for this project:

```
pip install pandas matplotlib nltk scikit-learn ipython
```

### 2. Loading and Preprocessing the Data
#### 2.1. Importing Libraries

```python
# Essential for data manipulation and analysis.
import pandas as pd

# Specific function from pandas for flattening JSON objects into a flat table.
from pandas import json_normalize

# Visualization tool for creating plots and graphs.
import matplotlib.pyplot as plt

# For parsing JSON data.
import json

# Toolkit for natural language processing and text analysis.
import nltk

# List of common words to filter out from text.
from nltk.corpus import stopwords

# Function to split text into individual words (tokens).
from nltk.tokenize import word_tokenize

# Transform texts into a suitable format for analysis.
from sklearn.feature_extraction.text import TfidfVectorizer

# Unsupervised machine learning algorithm for clustering.
from sklearn.cluster import KMeans

# Measures how similar an object is to its own cluster compared
# to other clusters.
from sklearn.metrics import silhouette_score

# For topic modeling.
from sklearn.decomposition import NMF

# Provides regular expression matching operations.
import re

# For interacting with the operating system.
import os

# For richer output formatting in Jupyter Notebooks.
from IPython.display import display, HTML

# Downloading the necessary NLTK datasets.
nltk.download('punkt')  # Tokenizer for breaking text into individual words.
nltk.download('stopwords')  # Common words to filter out from text.

pd.set_option('display.max_rows', None)  # To display all DataFrame rows.
pd.set_option('display.max_columns', None)  # To display all DataFrame columns.

# Setting max column width to 1000 characters.
pd.options.display.max_colwidth = 1000
```

Output:

```
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\lana\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\lana\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
```

This output indicates that the required NLTK datasets (`punkt` and `stopwords`) are being downloaded to the specified directory.

#### 2.2. Loading the Dataset

```python
# Initializing an empty list to store each parsed JSON object.
data = []

# Initializing a string to collect pieces of JSON spread across multiple lines.
partial_json = ""

# Opening and reading the dataset file.
with open('chatMessage.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            # Attempting to parse the line with any previously collected
            # JSON fragments.
            parsed_json = json.loads(partial_json + line)

            # If parsing is successful, appending the JSON object to the list
            # and resetting partial_json.
            data.append(parsed_json)

            # Clearing the partial_json to start fresh for the next object.
            partial_json = ""
        except json.JSONDecodeError:
            # If JSON decoding fails due to incomplete JSON fragments,
            # appending the line to partial_json to complete the JSON object.
            partial_json += line

# Loading the collected JSON objects into a DataFrame.
df = pd.DataFrame(data)

# Displaying the first few rows of the DataFrame to ensure data is loaded properly.
df.head()
```

Output:

|  | Item |
|---:|---:|
| 0 | {'id': {'S': 'd9b63dde-9a3b-4ba4-8289-d50e13eaf925'}, 'contents': {'S': '[PlayerName], is that person in route to your location? We have a team ready to enter your server, we just need you to accept the friend request from [PlayerName].'}, 'created': {'S': '2024-01-19T05:33:44.977Z'}, 'emergencyId': {'S': 'fdf41753-302f-4f70-87e2-070fe49c69c7'}, 'senderId': {'S': 'f2c0932a-2bb9-4edc-beeb-157cffd241ae'}, 'messageSentTimestamp': {'N': '1705642425'}} |
| 1 | {'id': {'S': 'bbe0090c-2a61-4989-8832-826a966bc58e'}, 'contents': {'S': 'Totally understandable'}, 'created': {'S': '2024-03-20T00:21:16.388Z'}, 'emergencyId': {'S': 'b22bc33c-7859-4203-9798-7d87e23d9866'}, 'senderId': {'S': 'ae859e9a-72e8-41c2-9e2e-97d7b7fab0a9'}, 'messageSentTimestamp': {'N': '1710894076'}} |
| 2 | {'id': {'S': 'db9a38e3-19e7-4037-b696-f9c4eaa278e3'}, 'contents': {'S': 'This emergency was submitted via the __**Client Portal**__'}, 'created': {'S': '2023-12-20T17:55:28.476Z'}, 'emergencyId': {'S': '7e6b13eb-5293-4a5e-803b-1bce8498edbd'}, 'senderId': {'S': 'c4859bd5-2600-4429-ade7-55861a1fa10a'}, 'messageSentTimestamp': {'N': '1703094928'}} |
| 3 | {'id': {'S': 'daeefb35-4cbe-4a3d-966f-6e9fa475cb63'}, 'contents': {'S': 'Hello! Thank you for choosing [Organization]. I will be sending a friend request, followed by a party invite. To accept a party and friend invite in an incapacitated state, press the "[" (Left Bracket) Button, which is located to the right of the "P" Button, as soon as you see a notification pop up in the middle of your screen. Please let me know when you are ready in and in first-person to accept.'}, 'created': {'S': '2024-01-22T22:06:15.605Z'}, 'emergencyId': {'S': 'e655cadf-9bd9-474f-94dd-9cbecba1e296'}, 'senderId': {'S': 'b3751979-6cb1-42ff-a1c6-f12d86499d99'}, 'messageSentTimestamp': {'N': '1705961176'}} |
| 4 | {'id': {'S': '4db6b8b4-2222-4ee3-ad30-7c1e1ac7bf57'}, 'contents': {'S': 'npc production is high'}, 'created': {'S': '2024-03-23T21:44:36.788Z'}, 'emergencyId': {'S': '9375d1de-337b-4f74-a784-07c73a791d0f'}, 'senderId': {'S': '2038299e-a532-41b1-b605-1085cea5e747'}, 'messageSentTimestamp': {'N': '1711230277'}} |

Our JSON object reveals to have a nested structure.

```python
# Normalizing nested JSON data from the "Item" column to create a flat
# table structure.
df_normalized = json_normalize(df['Item'])

# Combining the normalized data back with the original DataFrame,
# and dropping the original "Item" column which contained nested JSON.
chat = pd.concat([df.drop('Item', axis=1), df_normalized], axis=1)

# Displaying the first five rows of the processed DataFrame to verify
# that it loaded and normalized correctly.
chat.head()
```

Output:

|  | id.S | contents.S | created.S | emergencyId.S | senderId.S | messageSentTimestamp.N |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | d9b63dde-9a3b-4ba4-8289-d50e13eaf925 | [PlayerName], is that person in route to your location? We have a team ready to enter your server, we just need you to accept the friend request from [PlayerName]. | 2024-01-19T05:33:44.977Z | fdf41753-302f-4f70-87e2-070fe49c69c7 | f2c0932a-2bb9-4edc-beeb-157cffd241ae | 1705642425 |
| 1 | bbe0090c-2a61-4989-8832-826a966bc58e | Totally understandable | 2024-03-20T00:21:16.388Z | b22bc33c-7859-4203-9798-7d87e23d9866 | ae859e9a-72e8-41c2-9e2e-97d7b7fab0a9 | 1710894076 |
| 2 | db9a38e3-19e7-4037-b696-f9c4eaa278e3 | This emergency was submitted via the __**Client Portal**__ | 2023-12-20T17:55:28.476Z | 7e6b13eb-5293-4a5e-803b-1bce8498edbd | c4859bd5-2600-4429-ade7-55861a1fa10a | 1703094928 |
| 3 | daeefb35-4cbe-4a3d-966f-6e9fa475cb63 | Hello! Thank you for choosing [Organization]. I will be sending a friend request, followed by a party invite. To accept a party and friend invite in an incapacitated state, press the "[" (Left Bracket) Button, which is located to the right of the "P" Button, as soon as you see a notification pop up in the middle of your screen. Please let me know when you are ready in and in first-person to accept. | 2024-01-22T22:06:15.605Z | e655cadf-9bd9-474f-94dd-9cbecba1e296 | b3751979-6cb1-42ff-a1c6-f12d86499d99 | 1705961176 |
| 4 | 4db6b8b4-2222-4ee3-ad30-7c1e1ac7bf57 | npc production is high | 2024-03-23T21:44:36.788Z | 9375d1de-337b-4f74-a784-07c73a791d0f | 2038299e-a532-41b1-b605-1085cea5e747 | 1711230277 |

The JSON normalization process converts nested structures into a flat table by using the path to each element in the nested JSON as the column name. The suffixes in each column name, such as `.S` in `id.S`, `contents.S`, `created.S`, etc., and `.N` in `messageSentTimestamp.N`, indicate the data type or structure from the original JSON object. Here, `.S` stands for a string type, and `.N` in `messageSentTimestamp.N` denotes a numerical type.

#### 2.3. Renaming Columns

To improve the readability and usability of our dataset within the analysis, I will remove the `.S` and `.N` suffixes from each column name.

```python
# Renaming columns to more descriptive and simpler names.
chat = chat.rename(
    columns={
        'id.S': 'ID',
        'contents.S': 'contents',
        'created.S': 'created',
        'emergencyId.S': 'emergency ID',
        'senderId.S': 'sender ID',
        'messageSentTimestamp.N': 'message sent timestamp'
    }
)
```

#### 2.4. Changing Column Types

We begin by examining the current data types of the columns using the `info()` method.

```python
chat.info()
```
Output:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 16509 entries, 0 to 16508
Data columns (total 6 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   ID                      16509 non-null  object
 1   contents                16507 non-null  object
 2   created                 16509 non-null  object
 3   emergency ID            16509 non-null  object
 4   sender ID               16509 non-null  object
 5   message sent timestamp  16509 non-null  object
dtypes: object(6)
memory usage: 774.0+ KB
```

From the output of `chat.info()`, we can observe the structure of our DataFrame: it contains 6 columns and 16&nbsp;509 rows.

This output also reveals that all columns are currently recognized as `object` type, which is a generic type for storing data in pandas.

The `created` and `message sent timestamp` columns are currently formatted as objects but represent date-time information. The `created` column appears in ISO 8601 format (example value: `2024-03-23T21:44:36.788Z`), while `message sent timestamp` uses a Unix timestamp format (example value: `1711230277`). I will convert both to pandas datetime objects for consistency.

```python
# Converting "created" from ISO 8601 format string to datetime.
chat['created'] = pd.to_datetime(chat['created'])

# Ensuring the "created" datetime column is timezone-unaware (naive).
chat['created'] = chat['created'].dt.tz_localize(None)

# Flooring the "created" datetime to the nearest second to remove any
# smaller time units.
chat['created'] = chat['created'].dt.floor('s')

# Converting the "message sent timestamp" from Unix timestamp to a proper
# datetime format.

# First, ensuring the column is treated as an integer for accurate
# datetime conversion.
chat['message sent timestamp'] = chat['message sent timestamp'].astype(int)

# Converting the integer timestamps in "message sent timestamp" to datetime
# using Unix epoch time (seconds since 1970-01-01).
chat['message sent timestamp'] = pd.to_datetime(
    chat['message sent timestamp'], unit='s'
)
```

From our earlier call to `chat.info()`, we identified two missing values in the `contents` column. We will remove these rows and then convert the `contents` column to a string type to ensure compatibility with text processing functions in pandas.

```python
# Removing rows with missing values in the "contents" column.
chat = chat.dropna(subset=['contents'])

# Converting "contents" to string type.
chat['contents'] = chat['contents'].astype(str)

# Displaying the DataFrame info again to show the changes.
chat.info()
```

Output:

```
<class 'pandas.core.frame.DataFrame'>
Index: 16507 entries, 0 to 16508
Data columns (total 6 columns):
 #   Column                  Non-Null Count  Dtype         
---  ------                  --------------  -----         
 0   ID                      16507 non-null  object        
 1   contents                16507 non-null  object        
 2   created                 16507 non-null  datetime64[ns]
 3   emergency ID            16507 non-null  object        
 4   sender ID               16507 non-null  object        
 5   message sent timestamp  16507 non-null  datetime64[ns]
dtypes: datetime64[ns](2), object(4)
memory usage: 902.7+ KB
```

These conversions ensure that each column is optimized for the most appropriate and efficient data type.

The columns containing ID information, such as `ID`, `emergency ID`, and `sender ID`, will not be used in the analyses planned for this project, and therefore will be left in their original format.

#### 2.5. Removing System-generated Messages

Now, let's sort our DataFrame by `created` and take a look at the first registered chat messages.

```python
chat = chat.sort_values(by='created')
chat.head()
```

Output:

|  | ID | contents | created | emergency ID | sender ID | message sent timestamp |
|---:|---:|---:|---:|---:|---:|---:|
| 15824 | e525fec6-a67b-4c27-a4ba-dea204950ed7 | This emergency was submitted via the __**Client Portal**__ | 2023-06-18 16:19:22 | 647b34dd-cec1-4139-966a-e69ea5a3a2d2 | 27cae4ed-4217-4bf3-a5dd-d026667e7427 | 2023-06-18 16:19:22 |
| 10484 | 121bab33-c796-43ef-abdf-6a09d32f3e79 | This emergency was submitted via the __**Client Portal**__ | 2023-06-18 18:02:31 | ad77b83f-703c-4410-a2bc-5f4e4aa0b67e | a1e75072-8e7d-402c-93d6-6f48b189c090 | 2023-06-18 18:02:31 |
| 7560 | 25dc0a48-3a1b-457a-9e70-0b127da8e7e1 | This emergency was submitted via the __**Client Portal**__ | 2023-06-19 23:04:09 | 50c13e59-8511-4911-9580-20a45e8ebaa6 | 93306d79-93b8-4ca9-bfe4-80b89408d788 | 2023-06-19 23:04:09 |
| 6855 | d6dbe7a7-2d44-4b7a-9b1e-398060165c75 | This emergency was submitted via the __**Client Portal**__ | 2023-06-20 02:39:15 | 0e349ed4-c74f-441a-8399-716b46fffa17 | 5ac7729f-8759-4e26-9528-e3654a45a763 | 2023-06-20 02:39:15 |
| 7679 | 29c0f03b-dce3-4c6e-9daa-68054a5331bf | This emergency was submitted via the __**Client Portal**__ | 2023-06-20 03:02:27 | 20fed128-3d60-4151-8a12-62b2152d7532 | 26a32e20-20e0-491d-b024-7af52e0e1e32 | 2023-06-20 03:02:27 |

The first messages seem to be system-generated, and I'm going to remove them.

From now on, for the purpose of this project I'm going to focus on `contents` and `created` columns.

```python
# Retaining only the "contents" and "created" columns.
chat = chat[['contents', 'created']]

# Removing the system-generated messages.

# Stripping leading and trailing whitespaces in the "contents" column first.
chat['contents'] = chat['contents'].str.strip()

# Filtering out rows containing the automated message.
chat = chat[
    ~chat['contents'].str.contains(
        "This emergency was submitted via the __\\*\\*Client Portal\\*\\*__",
        regex=True,
        na=False
    )
]

chat.head()
```

Output:

|  | contents | created |
|---:|---:|---:|
| 2033 | ## Emergency details from Client\n\n __The client's situation is:__ **Unconscious**\n\n __The client is located:__ **[Location]**\n\n __Is the client injured:__ **Yes**\n \n &gt; 2 and 3\n \n __Has the client sent an IG beacon:__ **No**\n\n __Is the client in a team?__ **No**\n \n __Are there enemies nearby?__ **Yes**\n \n __Remarks:__\n\n &gt; 17mins til death | 2023-07-08 01:39:32 |
| 9732 | ## Emergency details from Client\n\n __The client's situation is:__ **Unknown**\n\n __The client is located:__ **Unknown**\n\n __Is the client injured:__ **Yes**\n \n &gt; Tier 3 chest\n \n __Has the client sent an IG beacon:__ **No**\n\n __Is the client in a team?__ **No**\n \n __Are there enemies nearby?__ **Yes**\n \n __Remarks:__\n\n &gt; In a cave on [Location], PVP player standing over my body. | 2023-07-08 08:35:19 |
| 7290 | ## Emergency details from Client\n\n __The client's situation is:__ **Unconscious**\n\n __The client is located:__ **Cave on [Location]**\n\n __Is the client injured:__ **Yes**\n \n &gt; Tier 3 chest inj\n \n __Has the client sent an IG beacon:__ **No**\n\n __Is the client in a team?__ **No**\n \n __Are there enemies nearby?__ **Yes**\n \n &gt; 1 player\n \n __Remarks:__\n\n &gt; None | 2023-07-08 08:50:19 |
| 13750 | ## Emergency details from Client\n\n __The client's situation is:__ **Unconscious**\n\n __The client is located:__ **Near [Location] in space aboard my [Ship]**\n\n __Is the client injured:__ **No**\n \n __Has the client sent an IG beacon:__ **No**\n\n __Is the client in a team?__ **No**\n \n __Are there enemies nearby?__ **No**\n \n __Remarks:__\n\n &gt; Not a particularly pressing alert, if too many come in put me on a waiting list, I have 1 hour and 25 minutes left | 2023-07-08 18:38:59 |
| 1279 | ## Emergency details from Client\n\n __The client's situation is:__ **Stranded**\n\n __The client is located:__ **26km from [Location]**\n\n __Is the client injured:__ **No**\n \n __Has the client sent an IG beacon:__ **No**\n\n __Is the client in a team?__ **Yes**\n \n &gt; [PlayerName]; [PlayerName]\n \n __Are there enemies nearby?__ **No**\n \n __Remarks:__\n\n &gt; Teammates on [Location] struggling to stay alive, cant come pick me up.\nI understand if too close to [Location] und too dangerous to come pick me up :D | 2023-07-08 20:06:53 |

Now we see a different kind of system-generated messages in the first rows. Let's remove them as well.

```python
# Filtering out rows that start with "## Emergency details from Client".
chat = chat[
    ~chat['contents'].str.startswith(
        "## Emergency details from Client", na=False
    )
]

chat.head(20)
```

Output:

|  | contents | created |
|---:|---:|---:|
| 3184 | Test string. Dont mind. | 2023-09-12 17:50:24 |
| 2181 | okay | 2023-09-13 12:32:04 |
| 1493 | lol | 2023-09-13 12:32:11 |
| 12388 | I like [Drink] btw | 2023-09-13 12:33:22 |
| 3115 | no one cares | 2023-09-13 12:33:48 |
| 7151 | Well thats not very Client-Orientated business acumen is it :/ | 2023-09-13 12:34:32 |
| 885 | Be advised security on site - bottom floor | 2023-09-13 12:36:51 |
| 9642 | copy | 2023-09-13 12:37:04 |
| 11873 | your [Ship] is destroyed | 2023-09-13 12:41:29 |
| 16258 | Thought so | 2023-09-13 12:48:00 |
| 3249 | I am in a Bunker | 2023-09-13 17:51:15 |
| 12418 | Yes | 2023-09-13 17:51:26 |
| 7874 | Second Wave with 5 enemies remaining | 2023-09-13 17:51:46 |
| 14845 | Copy that | 2023-09-13 17:51:53 |
| 16216 | Thank you very much. Are there any Security guards left? | 2023-09-13 17:55:11 |
| 6596 | All are dead I think | 2023-09-13 17:55:32 |
| 6490 | Alright, thanks. We will be with you shortly | 2023-09-13 17:56:21 |
| 6809 | Hallo, wie geht es so? | 2023-09-13 21:50:34 |
| 8636 | Well, dead. lol | 2023-09-13 21:50:55 |
| 6328 | Hab euch beiden ne Party invite geschickt :D | 2023-09-13 21:51:17 |

These messages are human-written.

Let's see how many messages we have in our DataFrame now.

```python
len(chat)
```

Output:

```
13107
```

#### 2.6. Exploratory Attempt to Filter Non-English Messages

We could notice that not all messages were in English. To address this, I attempted to filter them out using the `langdetect` and `langid` libraries. 

`langdetect` identified 2&nbsp;742 messages as non-English, while `langid` detected 166 high-confidence messages in other languages. However, in both cases, most of the labeled messages were actually written in English. Consequently, I decided to discard the idea of filtering by language for our dataset.

Here is the code used for these attempts:

```
pip install langdetect langid
```

```python
from langdetect import detect, LangDetectException
import langid

# Function to detect the language using langdetect.
def detect_language_langdetect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

# Function to detect the language using langid and return both language code
# and confidence.
def detect_language_langid(text):
    lang, confidence = langid.classify(text)
    return lang, confidence

# Applying langdetect.
chat['language_langdetect'] = chat['contents'].apply(detect_language_langdetect)

# Applying langid.
chat[['language_langid', 'langid_confidence']] = chat['contents'].apply(
    lambda x: pd.Series(detect_language_langid(x))
)

# Filtering messages detected as non-English by langdetect.
non_english_langdetect = chat[chat['language_langdetect'] != 'en']

# Filtering messages detected as non-English with high confidence by langid.
high_confidence_threshold = 0.9
high_confidence_non_english_langid = chat[
    (chat['language_langid'] != 'en')
    & (chat['langid_confidence'] >= high_confidence_threshold)
]

# Displaying the count of non-English messages detected.
print(f"langdetect detected {len(non_english_langdetect)} "
      "non-English messages.")
print(f"langid detected {len(high_confidence_non_english_langid)} "
      "high-confidence non-English messages.")

# Displaying the top 10 detected non-English messages.
top_10_non_english_langdetect = non_english_langdetect.head(10)
display(top_10_non_english_langdetect[['contents', 'language_langdetect']])
top_10_non_english_langid = high_confidence_non_english_langid.head(10)
display(
    top_10_non_english_langid[
        ['contents', 'language_langid', 'langid_confidence']
    ]
)

# Cleaning up.
chat.drop(
    columns=['language_langdetect', 'language_langid', 'langid_confidence'],
    inplace=True
)
```

Output:

```
langdetect detected 2763 non-English messages.
langid detected 166 high-confidence non-English messages.
```

|  | contents | language_langdetect |
|---:|---:|---:|
| 2181 | okay | fi |
| 1493 | lol | es |
| 12388 | I like [Drink] btw | hr |
| 3115 | no one cares | it |
| 9642 | copy | it |
| 11873 | your [Ship] is destroyed | fr |
| 3249 | I am in a Bunker | de |
| 12418 | Yes | tr |
| 6809 | Hallo, wie geht es so? | de |
| 8636 | Well, dead. lol | es |

|  | contents | language_langid | langid_confidence |
|---:|---:|---:|---:|
| 8636 | Well, dead. lol | es | 4.879470 |
| 9495 | roger roger | da | 1.140495 |
| 10032 | no problem | it | 2.727206 |
| 9367 | no beacon | it | 2.727206 |
| 254 | so cold... | es | 1.500048 |
| 2399 | tresspassing 20 secs | fr | 1.276789 |
| 15482 | Hello! I am ready | de | 1.043444 |
| 13189 | dohh, portal | es | 2.330713 |
| 2167 | excellent | de | 1.700788 |
| 14149 | im stuck | de | 2.070063 |

#### 2.7. Text Preprocessing

To prepare the chat messages for analysis, I perform several preprocessing steps on the text data. These steps include removing punctuation, converting all text to lowercase, and eliminating common stop words (like "the", "is", etc.) that are not useful for identifying key themes.

Additionally, tokenization is part of this process. It breaks the text into individual words or phrases, allowing for more granular analysis.

```python
# Function for text preprocessing.
def preprocess(text):
    # Converting to lowercase.
    text = text.lower()
    # Tokenizing text.
    tokens = word_tokenize(text)
    # Removing stopwords.
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Removing non-alphabetic tokens.
    tokens = [word for word in tokens if word.isalpha()]
    return " ".join(tokens)

# Applying preprocessing to the "contents" column.
chat['processed_contents'] = chat['contents'].apply(preprocess)

# Displaying the first 20 rows of the DataFrame to verify the preprocessing.
chat.head(20)
```

Output:

|  | contents | created | processed_contents |
|---:|---:|---:|---:|
| 3184 | Test string. Dont mind. | 2023-09-12 17:50:24 | test string dont mind |
| 2181 | okay | 2023-09-13 12:32:04 | okay |
| 1493 | lol | 2023-09-13 12:32:11 | lol |
| 12388 | I like [Drink] btw | 2023-09-13 12:33:22 | like [Drink] btw |
| 3115 | no one cares | 2023-09-13 12:33:48 | one cares |
| 7151 | Well thats not very Client-Orientated business acumen is it :/ | 2023-09-13 12:34:32 | well thats business acumen |
| 885 | Be advised security on site - bottom floor | 2023-09-13 12:36:51 | advised security site bottom floor |
| 9642 | copy | 2023-09-13 12:37:04 | copy |
| 11873 | your [Ship] is destroyed | 2023-09-13 12:41:29 | [Ship] destroyed |
| 16258 | Thought so | 2023-09-13 12:48:00 | thought |
| 3249 | I am in a Bunker | 2023-09-13 17:51:15 | bunker |
| 12418 | Yes | 2023-09-13 17:51:26 | yes |
| 7874 | Second Wave with 5 enemies remaining | 2023-09-13 17:51:46 | second wave enemies remaining |
| 14845 | Copy that | 2023-09-13 17:51:53 | copy |
| 16216 | Thank you very much. Are there any Security guards left? | 2023-09-13 17:55:11 | thank much security guards left |
| 6596 | All are dead I think | 2023-09-13 17:55:32 | dead think |
| 6490 | Alright, thanks. We will be with you shortly | 2023-09-13 17:56:21 | alright thanks shortly |
| 6809 | Hallo, wie geht es so? | 2023-09-13 21:50:34 | hallo wie geht es |
| 8636 | Well, dead. lol | 2023-09-13 21:50:55 | well dead lol |
| 6328 | Hab euch beiden ne Party invite geschickt :D | 2023-09-13 21:51:17 | hab euch beiden ne party invite geschickt |

### 3. Vectorization

In this step, we transform the text data into a numerical format suitable for machine learning algorithms. We achieve this using TF-IDF (Term Frequency-Inverse Document Frequency), which helps in representing the importance of words in the context of the entire dataset.

```python
# Initializing the TF-IDF Vectorizer.
vectorizer = TfidfVectorizer()

# Fitting the vectorizer to the processed text data and transforming the text
# into numerical format.

# "X" will be a sparse matrix where each row represents a text message
# and each column represents a term.
X = vectorizer.fit_transform(chat['processed_contents'])
```

### 4. Initial Analysis
#### 4.1. Clustering

In this section, I will use K-means clustering to group similar messages. This technique can help identify common themes or frequently discussed topics.

The advantage of clustering is that it can group messages that are semantically similar even if they don't use the exact same words.

I begin by retrieving the total number of CPU cores available on my PC to determine the computational resources available for parallel processing.

```python
print(os.cpu_count())
```

Output:

```
12
```

```python
# Setting the maximum number of CPU cores to be used by joblib to 8.
# Joblib is a library for parallel computing in Python, used by scikit-learn
# to speed up computations.
os.environ['LOKY_MAX_CPU_COUNT'] = '8'

# Starting with the number of clusters equal to 5.
k = 5

# Initializing the KMeans clustering algorithm with "k" clusters.
kmeans = KMeans(n_clusters=k, random_state=0)

# Fitting the KMeans model to the data and predicting cluster assignments.
clusters = kmeans.fit_predict(X)

# Attaching the predicted cluster labels back to the original DataFrame.
chat['cluster'] = clusters
```

Next, I will display the texts within each cluster to be able to identify common themes.

To add perspective, I will calculate the size of each cluster and its percentage relative to the entire pool of messages.

```python
# Calculating cluster sizes and percentages.
cluster_counts = chat['cluster'].value_counts()
total_counts = len(chat)
cluster_percentages = ((cluster_counts / total_counts) * 100).round(2)

# Sorting clusters by size.
sorted_cluster_indices = cluster_counts.sort_values(ascending=False).index

# Displaying clusters, sorted by cluster size.
for i in sorted_cluster_indices:
    # Sampling texts from each cluster.
    sample = chat[chat['cluster'] == i]['contents'].sample(7).to_frame()
    
    # Getting size and percentage of the current cluster.
    cluster_size = cluster_counts.loc[i]
    cluster_percentage = cluster_percentages.loc[i]
    
    # Formatting the header with size and percentage, and displaying
    # the sample texts.
    display(HTML(f"<h3>Cluster {i}: Size = {cluster_size}, "
                 f"Percentage = {cluster_percentage}%</h3>"))
    display(HTML(sample.to_html(escape=False)))
```

Output:

**Cluster 0: Size = 10775, Percentage = 82.21%**

|  | contents |
|---:|---:|
| 4334 | We certainly can come get you :) |
| 14029 | A Team is finally freed up, and will be joining this message thread momentarily. |
| 2816 | We're going to see if we can get to your body and pull it out |
| 3300 | Tested party chat, it's not working on my end. |
| 6315 | would you please fill out the additional info on the side |
| 4490 | just to confirm here |
| 3990 | 200 meters |

**Cluster 3: Size = 1242, Percentage = 9.48%**

|  | contents |
|---:|---:|
| 3652 | I have sent the friend request |
| 5241 | Invite sent |
| 1114 | im sending you a friend request and a party invite, please accept both |
| 4485 | Party invite sent. Please press the Left Bracket key `[` |
| 7006 | im gonna send you a party invite now |
| 13447 | Sending party invite now. |
| 880 | Hello! Thank you for choosing [Organization]. I will be sending a friend request, followed by a party invite. To accept a party and friend invite in an incapacitated state, press the "[" (Left Bracket) Button, which is located to the right of the "P" Button, as soon as you see a notification pop up in the middle of your screen. !! Please let me know when you are ready in and in first-person to accept !!. |

**Cluster 2: Size = 506, Percentage = 3.86%**

|  | contents |
|---:|---:|
| 9856 | Greetings! Thank You For Choosing [Organization] Services. Your Alert Has Been Received, Please Stand By To Accept A Friend Request And A Party Invite From The Team Lead. (To Accept The Invitations, You Will Need To Be In 1st Person And You Will Need To Press The Left Bracket Key `[` To Accept. Team Lead Will Notify You When They Are Sending The Invites.) |
| 11275 | Thank you for choosing [Organization], your alert has been received. A [Organization] team will be deployed to assist you shortly, please stand by to accept a friend request and a party invite from our Team Lead. (To accept the invitations, You should follow the Team Leader Instructions) * *The Team Lead assigned, will inform you, when they are sending their invites.* After you joined the party, please stand by to answer a few follow-up questions, so that we can provide a better service. |
| 15376 | Thank you for choosing [Organization], your alert has been received. A [Organization] team will be deployed to assist you shortly, please stand by to accept a friend request and a party invite from our Team Lead. (To accept the invitations, You should follow the Team Leader Instructions) * *The Team Lead assigned, will inform you, when they are sending their invites.* After you joined the party, please stand by to answer a few follow-up questions, so that we can provide a better service. |
| 3175 | Thank you for choosing [Organization], your alert has been received. A [Organization] team will be deployed to assist you shortly, please stand by to accept a friend request and a party invite from our Team Lead. (To accept the invitations, You should follow the Team Leader Instructions) * *The Team Lead assigned, will inform you, when they are sending their invites.* After you joined the party, please stand by to answer a few follow-up questions, so that we can provide a better service. |
| 6816 | Your alert has been received! A [Organization] team will be deployed to assist you shortly. I will be sending you a friend request and then a party invite shortly. |
| 5411 | Greetings! Thank You For Choosing [Organization] Services. Your Alert Has Been Received, Please Stand By To Accept A Friend Request And A Party Invite From The Team Lead. (To Accept The Invitations, You Will Need To Be In 1st Person And You Will Need To Press The Left Bracket Key `[` To Accept. Team Lead Will Notify You When They Are Sending The Invites.) |
| 460 | Thank you for choosing [Organization], your alert has been received. A [Organization] team will be deployed to assist you shortly, please stand by to accept a friend request and a party invite from our Team Lead. (To accept the invitations, You should follow the Team Leader Instructions) * *The Team Lead assigned, will inform you, when they are sending their invites.* After you joined the party, please stand by to answer a few follow-up questions, so that we can provide a better service. |

**Cluster 4: Size = 344, Percentage = 2.62%**

|  | contents |
|---:|---:|
| 1544 | Let me know when you are ready to receive the invitiations :) |
| 3666 | ready |
| 9159 | Are you ready to accept the invite? |
| 10210 | Hello there, ready for a party invite? |
| 3301 | Can you let your friend know I am also sending them a party invite? Let me know when they are ready. |
| 11977 | Ready for invites? |
| 9353 | let me know when you are ready |

**Cluster 1: Size = 240, Percentage = 1.83%**

|  | contents |
|---:|---:|
| 11983 | sorry been busy inside the bunker |
| 825 | sorry about this im new to getting a [Organization] usually my buddy does it\ |
| 7512 | sorry for the hold up, elevators are broken |
| 12451 | sorry for this mess. This suit is extremely rare, I'd like to take it back |
| 15310 | yeah sorry about that. worst timing ever |
| 5845 | sorry, crash got you |
| 9796 | sorry i missed it, you can cancel this request. i think we have it handled now. thank you! |

We can notice that a large cluster, comprising almost 4% of the data, consists of another type of automatic messages. We will use this discovery later to remove those messages.

#### 4.2. Topic Modeling with NMF

Non-negative Matrix Factorization (NMF) is a topic modeling technique that can be used to discover the hidden thematic structure in large archives of text. Unlike K-means, NMF is a soft clustering method, meaning that each document can be associated with multiple topics, each with a certain weight.

Let’s apply NMF to our dataset and explore its effectiveness.

```python
# Using TF-IDF Vectorizer for NMF.
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(chat['processed_contents'])

# Applying NMF to the TF-IDF features.
nmf_model = NMF(n_components=5, random_state=1, init='nndsvd').fit(tfidf)

# Retrieving the feature names (words) from the TF-IDF vectorizer.
feature_names = tfidf_vectorizer.get_feature_names_out()

# Displaying topics and their key words.
for topic_idx, topic in enumerate(nmf_model.components_):
    print(f"Topic #{topic_idx}:")
    # Displaying top 10 words for each topic.
    print(" ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))
```

Output:

```
Topic #0:
team lead accept [Organization] stand please shortly party alert received
Topic #1:
invite party sent friend sending please request left bracket press
Topic #2:
ready know let receive invitiations invites hello im accept see
Topic #3:
copy thanks alright shortly route en way enroute still good
Topic #4:
thank choosing [Organization] thanks ok services safe stay yes much
```

#### 4.3. Evaluation of the First Results

**Evaluation of Clustering**

- **Largest Cluster (82%)**: This cluster primarily consists of miscellaneous messages that do not fit into other clusters.

- **Second Largest Cluster (9%)**: Predominantly centered around the words "send/sent/sending," this cluster mostly involves discussions about friend requests or party invites, both essential for the community's rescue services.

- **Third Largest Cluster (4%)**: This cluster reveals another type of pre-written message appearing in slightly different forms.

- **Second Smallest Cluster (3%)**: Messages in this cluster frequently use the word "ready", typically asking if a person is prepared to receive an invite.

- **Smallest Cluster (2%)**: This cluster is formed around the word "sorry", used in various contexts.

**Evaluation of Topic Modeling**

- **Topic #0**: Reflects the theme of the pre-written message discovered during clustering: "Alert received, please stand by, the Team Lead assigned".

- **Topic #1**: Centers on sending a friend request and a party invite, and asking the recipient to press the left bracket.

- **Topic #2**: Related to Topic #1, with phrases like "Let me know when you're ready to receive the invites".

- **Topic #3**: Appears to be a generic confirmation message: "Copy, thanks, alright".

- **Topic #4**: Resembles a goodbye message: "Thank you for choosing [Organization] services, stay safe".

**Summary and Next Steps**

While both approaches provide valuable insights, I found the clustering results easier to interpret. I will continue to work with clustering, adjusting the number of clusters to extract further insights.

### 5. Fine-Tuning Clustering

#### 5.1. Removing Identified Pre-written Messages

From the analysis of the clusters, it became apparent that there is an existing pre-written type of message:

> Thank you for choosing [Organization], your alert has been received.  A [Organization] team will be deployed to assist you shortly, please stand by to accept a friend request and a party invite from our Team Lead. (To accept the invitations, You should follow the Team Leader Instructions)  * *The Team Lead assigned, will inform you, when they are sending their invites.*  After you joined the party, please stand by to answer a few follow-up questions, so that we can provide a better service.

I will remove the rows containing this message before re-running the clustering algorithm. This adjustment will help focus on more variable and unique interactions, potentially revealing deeper insights.

```python
# Defining a regular expression pattern that accounts for possible variations.
# This regex will be flexible with spaces and punctuation.
pattern = (
    r"thank you for choosing [Organization], your alert has been received\..*"
    r"please stand by to accept a friend request and a party invite "
    r"from our team lead\."
)

# Using the regex pattern to filter out rows.
# The "flags=re.I" parameter makes the match case insensitive.
chat = chat[
    ~chat['contents'].str.contains(
        pattern, case=False, na=False, regex=True, flags=re.I
    )
]

# Checking how many messages remain in our DataFrame.
len(chat)
```

Output:

```
12685
```

#### 5.2. Determining the Optimal Number of Clusters

In this chapter, I will focus on finding the optimal number of clusters for our dataset. I will use two commonly applied methods to achieve this: the Elbow Method and the Silhouette Score Method.

**Elbow Method**

The Elbow Method helps to determine the optimal number of clusters by plotting the sum of squared distances from each point to its assigned cluster center (within-cluster sum of squares) against the number of clusters. The point at which the rate of decrease sharply slows down, forming an "elbow", suggests an optimal cluster count.

```python
# Initializing an empty list to store the Within-Cluster Sum of Squares
# (WCSS) values.
wcss = []

# Looping through a range of cluster numbers from 1 to 50.
for i in range(1, 50):
    # Initializing the KMeans clustering algorithm with "i" clusters.
    kmeans = KMeans(
        n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0
    )
    # Fitting the KMeans model to the data.
    kmeans.fit(X)  # "X" is my vectorized data from earlier steps.
    wcss.append(kmeans.inertia_)  # "inertia_" is the WCSS for the given model.

# Plotting the results to visualize the Elbow Method.
plt.plot(range(1, 50), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
```

Output:

![Figure 1 - The Elbow Method](https://github.com/lanavirsen/Analyzing-Chat-Messages-for-Future-Automation/blob/main/images/Figure1.png)

*Figure 1. The Elbow Method*

**Silhouette Score Method**

The Silhouette Score Method evaluates the quality of clusters by measuring how similar each point is to its own cluster compared to other clusters. Scores range from -1 to 1, with higher scores indicating better-defined clusters. By plotting the Silhouette Score against different numbers of clusters, we can identify the cluster count that maximizes this score, indicating the optimal clustering solution.

```python
# Initializing an empty list to store the silhouette scores.
silhouette_scores = []

# Defining the range of cluster numbers to evaluate, starting from 2
# because the silhouette score cannot be calculated for a single cluster.
K_range = range(2, 50)

# Looping through the range of cluster numbers.
for k in K_range:
    # Initializing the KMeans clustering algorithm with "k" clusters.
    kmeans = KMeans(n_clusters=k, random_state=10)
    # Fitting the KMeans model to the data and predicting cluster labels.
    cluster_labels = kmeans.fit_predict(X)
    # Calculating the average silhouette score for the current number of clusters.
    silhouette_avg = silhouette_score(X, cluster_labels)
    # Appending the silhouette score to the list.
    silhouette_scores.append(silhouette_avg)

# Plotting the silhouette scores to visualize the results.
plt.plot(K_range, silhouette_scores)
plt.title("Silhouette Score Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
```

Output:

![Figure 2 - Silhouette Score Method](https://github.com/lanavirsen/Analyzing-Chat-Messages-for-Future-Automation/blob/main/images/Figure2.png)

*Figure 2. Silhouette Score Method*

**Analyzing the Results**

The Elbow Method did not show a clear "elbow", even with an extended range of clusters.

Additionally, the highest Silhouette Score obtained was 0.1, which is generally considered low and indicates poorly separated clusters.

These findings suggest that the data might not naturally separate into distinct groups very well, or there could be a high degree of overlap in the structure of the data points (chat messages).

**Next Steps**

After iterative testing, where I ran the clustering algorithm several times with different numbers of clusters, I found that setting the number of clusters to **14** is optimal. I focused only on clusters containing more than 50 messages. A smaller number of clusters does not reveal more nuanced, yet still valuable themes, while a larger number of clusters starts "overfitting", grouping messages by frequently used but very abstract themes, such as the verbs "see" and "get".

### 6. Final Clustering Execution

#### 6.1. Running the Model

```python
# Re-vectorizing the "processed_contents" column after filtering out some rows
# in the earlier steps.
X = vectorizer.fit_transform(chat['processed_contents'])

# The chosen optimal number of clusters.
k = 14

# Initializing the KMeans clustering algorithm with "k" clusters.
kmeans = KMeans(n_clusters=k, random_state=0)

# Fitting the KMeans model to the data and predicting cluster assignments.
clusters = kmeans.fit_predict(X)

# Rewriting the existing "cluster" column with new clusters.
chat['cluster'] = clusters

# Calculating cluster sizes and percentages.
cluster_counts = chat['cluster'].value_counts()
total_counts = len(chat)
cluster_percentages = ((cluster_counts / total_counts) * 100).round(2)

# Sorting clusters by size.
sorted_cluster_indices = cluster_counts.sort_values(ascending=False).index

# Displaying clusters, sorted by cluster size.
for i in sorted_cluster_indices:
    # Getting size and percentage of the current cluster.
    cluster_size = cluster_counts.loc[i]
    cluster_percentage = cluster_percentages.loc[i]
    
    # Only displaying clusters with more than 50 messages.
    if cluster_size > 50:
        # Sampling texts from each cluster.
        sample = chat[chat['cluster'] == i]['contents'].sample(7).to_frame()
        
        # Formatting the header with size and percentage, and displaying
        # the sample texts.
        display(HTML(f"<h3>Cluster {i}: Size = {cluster_size}, "
                     f"Percentage = {cluster_percentage}%</h3>"))
        display(HTML(sample.to_html(escape=False)))
```

Output:

**Cluster 4: Size = 8398, Percentage = 66.2%**

|  | contents |
|---:|---:|
| 14315 | This is us with you |
| 4052 | Hello There [PlayerName], I notice you mention having 2 ships on site. How did you achieve that, if you don't mind my asking? |
| 8858 | Yes, 7 angry individuals await you. Inclunding a [Ship] one who resisted a full magazine of grenades^^ |
| 15226 | the server is full |
| 10991 | I think I got it |
| 7772 | Howdy |
| 6562 | so for the question things, there are like multiple pages, so plase make sure that you fill all those things out until it says submit information |

**Cluster 9: Size = 778, Percentage = 6.13%**

|  | contents |
|---:|---:|
| 14781 | Friend request sent |
| 1778 | Please be on standby and **in First Person View** to accept a friend request and a party invite from our Team Lead; **[PlayerName]** *We will notify you when we are about to send the friend request and party invite!* |
| 6831 | Friend Request sent. Please press the Left Bracket key `[` |
| 5156 | Friend request sent |
| 5863 | I am going to send you a friend request then a subsequent party request so the team can locate and recover you. Please accept by pressing left bracket |
| 3669 | Friend Request sent. Please press the Left Bracket key `[` |
| 15629 | Hello [PlayerName]! I am the team lead! Be in first person for a friend request and party invite from [PlayerName] |

**Cluster 3: Size = 667, Percentage = 5.26%**

|  | contents |
|---:|---:|
| 467 | Hi [PlayerName], thank you for for choosing [Organization]. Our team is currently on the way back from another mission, I'll let you know when we're on the way. |
| 14329 | A Team will be on the way. |
| 9299 | Our [Organization] Team is en route to your location. We will be with you ASAP. |
| 15683 | Your server is incredibly unstable! A team is now on their way! |
| 12413 | I will have a team assigned to you ASAP, one is wrapping up. |
| 11643 | Team is inbound. |
| 6697 | Wonderful! Our Team Is Joining Your Session Now! |

**Cluster 6: Size = 573, Percentage = 4.52%**

|  | contents |
|---:|---:|
| 8018 | sending |
| 9220 | Sending you a party invite now [PlayerName], we'll see if we can't pick you up. |
| 734 | Sending a party invite now [PlayerName] |
| 16427 | Thank you for the information. I'll be sending party invite in 2 mins |
| 5736 | I send party invite now please accept thank you |
| 36 | Sending party invite. Please be in first-person to accept. |
| 13236 | Copy a party invite will be coming shortly so please get ready to accept it in game |

**Cluster 2: Size = 410, Percentage = 3.23%**

|  | contents |
|---:|---:|
| 3776 | As we conclude our service, we extend our sincere gratitude for choosing [Organization]. Your health and satisfaction are our top priorities. Should you ever require assistance again, we stand ready to provide our services. Safe travels and goodbye! |
| 2781 | Let me know when you are ready to receive the invitiations :) |
| 15613 | Let me know when you ready for the invites :) |
| 13350 | As we conclude our service, we extend our sincere gratitude for choosing [Organization]. We hope what we provided today was prompt, professional, and met your expectations! Your health and satisfaction are our top priorities. We value your trust and hope you choose us again should you need to! If you find yourself requiring assistance again, we stand ready to provide our services. |
| 5368 | Hello! Let me know when you are ready for invites :) |
| 2304 | Let me know when you are ready for the invtes |
| 14071 | Let me know when you are ready to receive the invitiations :) |

**Cluster 1: Size = 355, Percentage = 2.8%**

|  | contents |
|---:|---:|
| 5968 | Thank you for choosing [Organization] services, have fun and have great day! |
| 7837 | Thank you for chosing [Organization] services! It was a pleasure serving you. Stay safe out there! |
| 721 | server full, loggin to another, thank you [Organization] o7 |
| 4635 | Thank you |
| 10404 | Thank you for your patience. We are in the server and we are departing. |
| 7127 | thank you the team is now on there way |
| 450 | Thank you for chosing [Organization] have a nice day :) |

**Cluster 7: Size = 279, Percentage = 2.2%**

|  | contents |
|---:|---:|
| 494 | i am very sorry to have missed that |
| 8625 | sorry for the late response are you still in need of assistance? |
| 5997 | I'm sorry for your loss, please cancel the alert on the portal. we will be more than hppy to provide services if needed. |
| 27 | I am sorry for that, thank you for choosing [Organization] and hope you have a good night. |
| 15371 | I also think the planet is [Location], sorry if I'm mistaken. |
| 5167 | As we conclude our service, we extend our sincere gratitude for choosing [Organization]. Your health and satisfaction are our top priorities. Should you ever require assistance again, we stand ready to provide our services. Safe travels and goodbye! If you would be so kind as to share your thoughts on the service provided, your input will help us continually enhance the quality of our offerings. To rate our service, please consider taking a moment to provide your feedback at the top of this page. |
| 16376 | "[" sorry |

**Cluster 13: Size = 270, Percentage = 2.13%**

|  | contents |
|---:|---:|
| 12758 | and I see you closer now |
| 14554 | I don't see him anymore. Not sure where he's gone. |
| 1275 | Did you see my last message? :) |
| 1419 | Switching comms over to Party Chat. Let me know if you are unable to see it. |
| 15390 | I do not see any |
| 8986 | Do you still see the players? Any sign or combat still? :) |
| 1857 | no PVE, that i could see |

**Cluster 11: Size = 265, Percentage = 2.09%**

|  | contents |
|---:|---:|
| 14410 | can you instruct your teammate to do so as well, thanks ahead |
| 5313 | Thanks for the heads up |
| 15089 | nah thanks ill be ok |
| 2321 | I'll go with my ship, thanks guys |
| 13251 | thanks for choosing [Organization] services! Fly safe! |
| 1215 | Thanks |
| 9903 | thanks anyway though |

**Cluster 0: Size = 207, Percentage = 1.63%**

|  | contents |
|---:|---:|
| 14917 | copy that |
| 7935 | copy |
| 9738 | [PlayerName] do you copy? |
| 16365 | copy that |
| 8189 | copy |
| 12854 | copy |
| 12361 | copy that |

**Cluster 5: Size = 162, Percentage = 1.28%**

|  | contents |
|---:|---:|
| 7532 | Ok sending now |
| 12383 | ok we have you |
| 5129 | Ok.. |
| 6163 | ok got you |
| 1246 | Ok, i'm no longer stranded |
| 2370 | no, my [Ship] is outside, i will be ok |
| 1372 | ah ok |

**Cluster 10: Size = 139, Percentage = 1.1%**

|  | contents |
|---:|---:|
| 15015 | Be advised: because you sent out a beacon, it's highly likely another player may reach you before we can. |
| 1900 | Beacon has been cancelled |
| 12870 | Unconscious, can't cancel my beacon. |
| 3890 | i see that someone accepted my beacon too |
| 5339 | if you do have a beacon, but I don't have it- can you tell dow does? |
| 2171 | we still have your beacon |
| 15953 | Back when I was solo beacon hunting I had a guy's marker start moving. Someone kidnapped him while downed. c_c |

**Cluster 8: Size = 104, Percentage = 0.82%**

|  | contents |
|---:|---:|
| 343 | accepted |
| 12599 | Received and accepted |
| 3969 | Accepted |
| 12141 | she has accepted |
| 1511 | I accepted |
| 6987 | I accepted |
| 4063 | Accepted |

**Cluster 12: Size = 78, Percentage = 0.61%**

|  | contents |
|---:|---:|
| 12872 | Roger -standing by |
| 8487 | Roger that |
| 4299 | Roger that |
| 3348 | Roger that |
| 2168 | Roger |
| 7647 | roger |
| 10102 | roger that. sending it now |

#### 6.2. Cluster Analysis

After filtering out system-generated and pre-written messages, the analysis revealed that a significant portion of the remaining chat messages - **66%** - couldn't be clustered in a meaningful way. However, several prominent themes emerged from the clustered messages:

- **778 messages** contained "Friend request sent".
- **667 messages** mentioned that "Team is on their way now".
- **573 messages** included "Sending party invite".
- **410 messages** stated "Let me know when you are ready for the invites".
- **355 messages** said "Thank you", with additional **265 messages** saying "Thanks".
- **279 messages** said "Sorry".
- **207 messages** said "Copy that", with additional **78 messages** saying "Roger that".
- **138 messages** mentioned "beacon".
- **104 messages** said "Accepted".

### 7. Project Conclusion and Recommendations

The most frequently used and lengthy messages that are written manually are related to sending friend requests and party invites. These steps occur at the very beginning of the alert process and are mandatory. The analysis indicates that automating these messages would significantly reduce the manual effort required by the team.

- Automating the message **"Friend Request sent. Please press the Left Bracket key `[`"** could streamline up to **6%** of the messages currently being written manually.

- Automating **"Sending party invite. Please be in first-person to accept"** could streamline up to **4.5%**.

- Automating **"Let me know when you are ready to receive the invitations"** could streamline up to **3.2%**.

Together, automating these three messages could streamline up to **13.7%** of the manually written messages.

Additionally, there are several short, frequently repeated messages that could also be considered for automation:

- **"Thank you"** - 2.8%
- **"Sorry"** - 2.2%
- **"Thanks"** - 2.1%
- **"Copy that"** - 1.6%
- **"Roger that"** - 0.6%

Assuming approximately 75% of these shorter messages are written by the team (and 25% by the client), automating them could streamline up to additional **7%** of the messages. Combined with the invitation-related messages, this could result in automating up to **20.7%** of the messages.

Depending on the available options for automation, prioritizing the invitation-related messages would provide the most significant benefit. However, if feasible, automating the shorter messages could further enhance communication efficiency.
