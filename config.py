from gensim.models import Word2Vec
w2vModel = Word2Vec.load("w2v-model/word2vec.model")
WeightVec=w2vModel.wv['100g']
VolumeVec=w2vModel.wv['100ml']
MAX_LEN = 36
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
EPOCHS=3
AGES = ['lasten','lapset','infant','infants','baby','toddler',
'children','child','newborn','newborns','junior','adult','adults','kids','kid']
GENDERS = ['women','men','man','girl','girls','boy','boys','unisex',
'male','tyttö','tytöt','poika','pojat',
'female','lady','wm','woman','naisten','nainen','mies','miesten']
COLORS = ['valkoinen','punainen','sininen','musta','ruskea',
'keltainen','oranssi','vihreä','vaaleanpunainen',
'harmaa','white','red','blue','black','brown','grey','orange',
'yellow','pink','tortoise','green','purple','multicolor','shiny',
'gold','metalli','cinnamon','light','dark',
'thunder','silver','havana','rose','alloy','indigo',
'mint','ivory','beige','pinkki','scarlet', 'lavendula',
'tan','charcoal','violet','graphene','oliivi','multi',
'kulta','vaaleanvihreä','hopea', 'transparent','lavender',
'olive','navy','snow','khaki','glitter','camo','taupe',
'bordeaux','copper','luonnonväri','denim','optic','väri']
TAG_TO_NUM = {'NIL': 0, 'O': 1, 'BRAND': 2, 'COLOR': 3,'GENDER': 4,'VOLUME':5,'WEIGHT':6,'AGE':7}
NUM_TO_TAG = {0:'NIL', 1:'O',2:'BRAND',3:'COLOR',4:'GENDER',5:'VOLUME',6:'WEIGHT',7:'AGE'}