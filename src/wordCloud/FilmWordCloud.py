import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

dataTrainPath = '../dataset/train_set.csv'
df = pd.read_csv(dataTrainPath, sep='\t')

stop = set(STOPWORDS)
stop.add("said")
stop.add("say")
stop.add("says")
stop.add("make")
stop.add("will")
stop.add("new")
stop.add("also")
stop.add("one")
stop.add("now")
stop.add("still")
stop.add("time")
stop.add("may")
cloud = WordCloud(background_color="white", max_words=2000, stopwords=stop)
positive_cloud = cloud.generate(df.loc[df['Category'] == 'Film', 'Content'].str.cat(sep='\n'))
plt.figure()
plt.imshow(positive_cloud)
plt.axis("off")
plt.show()