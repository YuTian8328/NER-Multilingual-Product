# NER & Extraction from product-data
The **goal** of this task is to extract product information such as **BRAND NAME, SIZE, COLOR, GENDER,AGE,VOLUME, WEIGHT** from product's titles and descriptions supplied by 119 unique providers. There are 670k observations in the dataset. There is no missing data in brand column, so it can be used to annotate Brand Name.  Meta column is also quite useful for labeling the data(size,color,gender and age) despite missing some information.

This is how it looks like
![sample](img/sample.png)

## Brand Names

<img src="img/brand.png"  align="left" width="20%"/>

Lots of brand names are rarely-used words or industry-created new words, thus existing word embedding methods, such as GloVe or Bert, can't properly embedding them. If these words can't be embedded properly in a NER model, it's definitely a big problem.


The solution is to train a custom word embedding model by using library **gensim** and library **nltk** with this special corpus. Based on this custom word embedding model, the NER Model can extract Brand Names quite effectively (Test set accuracy approximates 99.5%)

## COLOR, VOLUME, WEIGHT, GENDER, AGE
Color,volume,weight,gender and age are labled by using regex and information from meta column.

The NER Model has also achieved quite good performance in extracting color,volume and weight as well as brand name.
Test set accuracy is 99.65%



## SIZE relevant information
The forms of size relevant information varied: 
- "100cm"
- "3.0cm - 5.0cm"
- "95x220cm"
- "0,5 x 1,8 x 49cm"
- "kokoja: s, m, l, xl, xxl"
- "kokoja: us 34 us 35 us 36"
- "etupituus: 56cm, takapituus: 70cm"
- "R14"
- "kokoja: 3, 3 1/2, 4, 4 1/2"
- "kokoja: yksi koko"
- "koko: standard"
- ......

To extract this kind of information, NER is not a good choice (accuracy approximates 50% in our experiment). But utilizing Regex is a quite effective and efficient solution. Thus the final information extraction strategy will be a combination of regex and NER model.



## Work FLow

<img src="img/WorkFlow.jpg" width="600"/>

## NER model
A BiLSTM neural network is built to execute NER and Extraction.

The architecture looks like this:
<img src="img/architecture.png" width="700"/>

## Test With Unseen Text


```python
pd.set_option('display.max_colwidth', -1)
```

    /opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
      """Entry point for launching an IPython kernel.



```python
df.iloc[1:5,0:6]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>description</th>
      <th>summary</th>
      <th>brand</th>
      <th>price</th>
      <th>meta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Sc-Erna Polvipituinen Hame Sininen Soyaconcept</td>
      <td>SOYACONCEPT on tanskalainen brändi, joka luo elegantteja vaatteita romanttisilla yksityiskohdilla.. Takapituus: 55 cm.</td>
      <td>NaN</td>
      <td>Soyaconcept</td>
      <td>49.99</td>
      <td>{"SIZE": ["36"], "COLOR": ["cristal blue"], "GENDER": ["women"]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dana Buchman Silmälasit Taren CARAMEL TORTOISE</td>
      <td>Dana Buchman Taren Silmälasit. Collection:Men. Kehyksen Väri: Tortoise. Kehysmateriaali: Plastic. Koko: 54.</td>
      <td>NaN</td>
      <td>Dana Buchman</td>
      <td>146.00</td>
      <td>{"SIZE": ["54"], "COLOR": ["tortoise"], "GENDER": ["male"], "AGE_GROUP": ["adult"]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Active Sports Woven Shorts B Shortsit Musta PUMA</td>
      <td>PUMA Active Sports Woven Shorts B</td>
      <td>NaN</td>
      <td>PUMA</td>
      <td>27.00</td>
      <td>{"SIZE": ["164", "128", "110", "116", "104", "176"], "COLOR": ["puma black"], "GENDER": ["kids"]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Renata Polvipituinen Hame Musta Fall Winter Spring Summer</td>
      <td>Fall Winter Spring Summer. A-linjainen.</td>
      <td>NaN</td>
      <td>Fall Winter Spring Summer</td>
      <td>199.00</td>
      <td>{"SIZE": ["xs"], "COLOR": ["jet black"], "GENDER": ["women"]}</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python
# helper functiona
def print_row_data(df, loc):
    for k,v in df.iloc[loc].iteritems():
        print(f"{k}: {v}")
```


```python
print_row_data(df,81001)
```

    title: Dior Silmälasit DIOR ESSENCE 17 HT8
    description: Dior DIOR ESSENCE 17 Silmälasit. Collection:Women. Kehyksen Väri: Rose Tortoise. Kehysmateriaali: Plastic. Koko: 49.
    summary: nan
    brand: Dior
    price: 208.0
    meta: {"SIZE": ["49"], "COLOR": ["rose tortoise"], "GENDER": ["female"], "AGE_GROUP": ["adult"]}
    provider_category: 13-silmalasit-ja-piilolinssit
    provider: Smartbuy Glasses


# workflow


```python
a=['o',2,3]
b=a
```


```python
a.append(9)
```


```python
b=[1.2]
```


```python
b
```




    [1.2]




```python
a
```




    ['o', 2, 3, 9, 9]



<p style="font-size:20px;color:red;"> hhh </p>


```python

```
