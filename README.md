## SIMPLE CHAT BOT

### very simple chat bot - aked a question it will give back a predefined answer

#### Steps to make it work:
- Fork repo, including folder quant (in includes a small transformer).
- pip install -r requirements.txt.
- Open input and responses.csv file.
- Input potential question and associated answers, save the file.
- Run ```python chat.py True```
- First run will take a bit longer as it will create new embeddings file. For runs that do not need updated embeddings file, use ```python chat.py False```
- Default brower will open where you can ask questions and see predefined answers 

![image](https://user-images.githubusercontent.com/67911055/181908251-e1d34d09-08d6-4cd3-992f-3a468d5a9ec1.png)

