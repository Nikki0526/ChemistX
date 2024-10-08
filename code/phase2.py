# -*- coding: utf-8 -*-
"""phase2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12qmYG83HnN_mpt9GqoN2pgVWc6tNPJu2
"""

pip install openai==1.3.7

from google.colab import drive
drive.mount('/content/drive')

import os

os.chdir("/content/drive/MyDrive/Colab Notebooks/research/AI agent/100")

from openai import OpenAI

client = OpenAI(
    base_url="https://oneapi.xty.app/v1",
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

import re

file_list = []

def extract_and_save_python_code(text, name, repeated_times):

    # Regular expression pattern for extracting Python code blocks
    pattern = r"```python(.*?)```"
    # Use re.DOTALL to match across multiple lines
    matches = re.findall(pattern, text, re.DOTALL)

    # Save each extracted code block to a separate .py file
    for i, code_block in enumerate(matches, 1):
        file_name = f"extracted_code_block_{name}_{repeated_times}_{i}.py"
        with open(file_name, 'w') as file:
            file.write(code_block.strip())
        print(f"Saved Python code to {file_name}")

    file_list.append(file_name) # 保存python文件名

    return matches

def prompt_phase2():
  completion = client.chat.completions.create(
  model="gpt-4-0613",
  messages=[{"role": "system", "content": "You are a helpful assistant on python code generation and selenium webdriver."},
    {"role": "user", "content": '''I am using selenium webdriver to simulate operations on a website. Assume the WebDriver has already been initialized and the target URL has already been reached. Now I will explain the task to you step by step. Please help me generate python code to perform the task and note that the output should be a single .py file. The .py file contains a ‘run’ function which accepts a driver parameter. Please note that the initialized WebDriver is called ‘driver’ and you can use it as a parameter in ‘run’ function. You can use the XPath of the button to simulate clicking it and executing a JavaScript click. An example for the code is ‘’’
button = driver.find_element(By.XPATH, '//input[@type="checkbox" and @value="Product"]')
driver.execute_script("arguments[0].click();", button)’’’

The first step is to simulate clicking a button. The HTML for this button is ‘’’ <button _ngcontent...</button>‘’’

The second step is to simulate waiting 5 seconds and clicking the second button. The HTML for this button is ‘’’ <button _ngcontent...</button>‘’’

The third step is to simulate waiting 5 seconds and input ‘CC1=NC=CC(=C1)C2=CC3=C4C(=CN=C3C=C2)N=NN4C’ into the webpage. The HTML for the location of input is ‘’’<div class=...’’’

The fourth step is to simulate clicking the third button. The HTML for this button is ‘’’<button id...</button>’’’

The fifth step is to simulate waiting 5 seconds and clicking the fourth button. The HTML for this button is ‘’’<button class...</button>’’’

The sixth step is to simulate waiting 5 seconds and clicking the fifth button. The HTML for this button is ‘’’<button _ngcontent...</button>’’’

The seventh step is to simulate waiting 30 seconds and clicking the sixth button. The HTML for this button is ‘’’<ul class...’’’

The eighth step is to simulate waiting 15 seconds and downloading the html source code of this webpage. Please use UTF-8 encoding when writing to the file.

The ninth step is to simulate clicking another button. The HTML for this button is ‘’’ ... ‘’’

Please do not end the WebDriver session when the task is finished.


    '''}
  ],
  stream = True
)
  return completion

for i in range(1):
  completion = prompt_phase2()
  text_print = ""

  for chunk in completion:
    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
      text_print = text_print + chunk.choices[0].delta.content

  extracted_code_blocks = extract_and_save_python_code(text_print, 'phase2', i)

print(text_print)