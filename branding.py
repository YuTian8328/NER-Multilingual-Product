from preprocess import prepare_text
import json
import pandas as pd
from nltk.tokenize import word_tokenize
def extract_color(meta_str):
    if not pd.isna(meta_str):
        dict_ = json.loads(meta_str)
        if 'COLOR' in dict_:
            return ' '.join(dict_['COLOR'])
def labeling(label, string_chunk):
    return [(word, label) for word in word_tokenize(string_chunk)]
def branding(df):
    t, br = df.combined, df.brand
    start = (t.lower()).find(br.lower())
    end = start + len(br)
    labeled_text = labeling('O', t[:start]) + \
                    labeling('BRAND', t[start:end]) + labeling('O', t[end:])
    return labeled_text
if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    df['combined'] = df.title +', '+df.description.fillna(' ')
    df['color'] = df.meta.apply(extract_color)
    df.color.fillna(' ',inplace=True)
    df['origin_combined'] = df['combined'].values
    df['combined'] = df.origin_combined.apply(prepare_text)
    df['combined'] = df.apply(branding, axis=1)
    df.to_csv('after_branding.csv',index=False)