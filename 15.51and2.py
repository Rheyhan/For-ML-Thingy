#answer 1st question

#libraries needed
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np


if __name__ == "__main__":
    #We just gonna merge all of those things, cuz why not? Lol
    filenames=["night_and_day_virginia_woolf.txt", "the_way_of_all_flash_butler.txt", "moby_dick_melville.txt", "sons_and_lovers_lawrence.txt", "robinson_crusoe_defoe.txt", "james_joyce_ulysses.txt"]
    dummy=[]    #we put the dummy here
    for filename in filenames:
        yourfolderhere="files/"
        path=yourfolderhere+filename
        with open(path, encoding="utf8") as fh:    #can't do shit with jupiter cuh
            dummy.append(fh.read())

    #get model of the vectroizer
    vectorizer= text.CountVectorizer(stop_words=list(text.ENGLISH_STOP_WORDS))
    vectorizer.fit(dummy)
    token=vectorizer.transform(dummy)
    tokenarr=token.toarray()

    print(len(vectorizer.get_feature_names_out()))              #How many words are there on all of those texts combined
    print(tokenarr[:,1])
    #20 most common words used
    sortedbyoccurence=list(zip(vectorizer.get_feature_names_out(),tokenarr.sum(axis=0)))
    sortedbyoccurence.sort(key=lambda x:x[1], reverse=True)
    i=0;n=20
    print("20 words mostly used are:"); wordlist=[]; quantitylist=[]
    #aight screw this imma just use pandas instead cuz why not?
    for word, quantity in sortedbyoccurence:
        wordlist.append(word); quantitylist.append(quantity)
        if i>n:
            break
        i+=1
    funnydataframe=pd.DataFrame({"word": wordlist,
                                 "quantity": quantitylist})
    print(funnydataframe)